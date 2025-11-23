// --- 1. CONFIGURATION ---
const GRID_SIZE = 20;
const CELL_SIZE = 28;
const AGENT_NAMES = {
    'human': '👤 Human',
    'qlearning': '🤖 Q-Learning',
    'sarsa': '🤖 SARSA',
    'dqn': '🧠 DQN'
};

// --- 2. AGENTS ---
class BaseAgent {
    constructor() { this.epsilon = 1.0; this.decay = 0.995; this.minEpsilon = 0.01; }
    getStateVector(game) {
        const h = game.snake[0]; const f = game.food;
        const dirs = [{x: game.dir.y, y: -game.dir.x}, {x: game.dir.x, y: game.dir.y}, {x: -game.dir.y, y: game.dir.x}];
        const dangers = dirs.map(d => {
            const nx = h.x + d.x, ny = h.y + d.y;
            return (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE || game.snake.some(s => s.x === nx && s.y === ny)) ? 1 : 0;
        });
        const foodDir = [f.x < h.x ? 1:0, f.x > h.x ? 1:0, f.y < h.y ? 1:0, f.y > h.y ? 1:0];
        return { key: [...dangers, ...foodDir].join(''), vector: [...dangers, ...foodDir] };
    }
}

class QLearningAgent extends BaseAgent {
    constructor() { super(); this.qTable = {}; this.lr = 0.1; this.gamma = 0.9; }
    getAction(stateObj) {
        if(Math.random() < this.epsilon || !this.qTable[stateObj.key]) return Math.floor(Math.random()*4);
        const qs = this.qTable[stateObj.key]; const max = Math.max(...qs);
        const best = qs.map((v, i) => v === max ? i : -1).filter(i => i !== -1);
        return best[Math.floor(Math.random() * best.length)];
    }
    learn(state, action, reward, nextState, done) {
        if(!this.qTable[state.key]) this.qTable[state.key] = [0,0,0,0];
        if(!this.qTable[nextState.key]) this.qTable[nextState.key] = [0,0,0,0];
        const maxNext = Math.max(...this.qTable[nextState.key]);
        const cur = this.qTable[state.key][action];
        this.qTable[state.key][action] = cur + this.lr * (reward + this.gamma * maxNext - cur);
        if(done && this.epsilon > this.minEpsilon) this.epsilon *= this.decay;
    }
}

class SarsaAgent extends BaseAgent {
    constructor() { super(); this.qTable = {}; this.lr = 0.1; this.gamma = 0.9; }
    getAction(stateObj) {
        if(Math.random() < this.epsilon || !this.qTable[stateObj.key]) return Math.floor(Math.random()*4);
        const qs = this.qTable[stateObj.key]; const max = Math.max(...qs);
        return qs.map((v, i) => v === max ? i : -1).filter(i => i !== -1)[0];
    }
    learn(state, action, reward, nextState, done, nextAction) {
        if(!this.qTable[state.key]) this.qTable[state.key] = [0,0,0,0];
        if(!this.qTable[nextState.key]) this.qTable[nextState.key] = [0,0,0,0];
        const nextQ = this.qTable[nextState.key][nextAction] || 0;
        const cur = this.qTable[state.key][action];
        this.qTable[state.key][action] = cur + this.lr * (reward + this.gamma * nextQ - cur);
        if(done && this.epsilon > this.minEpsilon) this.epsilon *= this.decay;
    }
}

class DQNAgent extends BaseAgent {
    constructor() {
        super();
        this.model = null; this.memory = []; this.batchSize = 32; this.isReady = false;
        this.decay = 0.997; 
        this.initModel();
    }
    async initModel() {
        if(window.tf) {
            this.model = tf.sequential();
            this.model.add(tf.layers.dense({units: 24, inputShape: [7], activation: 'relu'}));
            this.model.add(tf.layers.dense({units: 24, activation: 'relu'}));
            this.model.add(tf.layers.dense({units: 4, activation: 'linear'}));
            this.model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
            this.isReady = true;
            console.log("DQN Model Loaded");
        }
    }
    getAction(stateObj) {
        if(!this.isReady || Math.random() < this.epsilon) return Math.floor(Math.random()*4);
        return tf.tidy(() => {
            const output = this.model.predict(tf.tensor2d([stateObj.vector]));
            return output.argMax(1).dataSync()[0];
        });
    }
    remember(s, a, r, ns, d) {
        this.memory.push({s: s.vector, a, r, ns: ns.vector, d});
        if(this.memory.length > 2000) this.memory.shift();
    }
    async replay() {
        if(this.memory.length < this.batchSize || !this.isReady) return;
        const batch = [];
        for(let i=0; i<this.batchSize; i++) batch.push(this.memory[Math.floor(Math.random()*this.memory.length)]);
        const states = tf.tensor2d(batch.map(x => x.s));
        const nextStates = tf.tensor2d(batch.map(x => x.ns));
        const currentQs = this.model.predict(states);
        const nextQs = this.model.predict(nextStates);
        const currentQsData = await currentQs.array();
        const nextQsData = await nextQs.array();
        for(let i=0; i<this.batchSize; i++) {
            const {a, r, d} = batch[i];
            currentQsData[i][a] = r + (d ? 0 : 0.9 * Math.max(...nextQsData[i]));
        }
        await this.model.fit(states, tf.tensor2d(currentQsData), {epochs: 1, verbose: 0});
        states.dispose(); nextStates.dispose(); currentQs.dispose(); nextQs.dispose();
        if(this.epsilon > this.minEpsilon) this.epsilon *= this.decay;
    }
}

// --- 3. CHART MGR ---
const chartMgr = {
    chart: null,
    compareMode: false,
    init() {
        const ctx = document.getElementById('learningChart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Init', data: [], borderColor: '#00f3ff', tension: 0.1, pointRadius: 0, borderWidth: 2 }] },
            options: { 
                responsive: true, maintainAspectRatio: false, 
                plugins: { legend: {display: true, labels: {color:'white', font:{family:'Orbitron'}}} },
                scales: { x: {display: false}, y: {grid: {color:'rgba(255,255,255,0.1)'}, ticks:{color:'#888'}} },
                animation: false
            }
        });
    },
    addPoint(ep, score) {
        const d = this.chart.data;
        d.labels.push(ep);
        const ds = d.datasets[d.datasets.length-1];
        ds.data.push(score);
        if(d.labels.length > 100) { d.labels.shift(); d.datasets.forEach(ds => { if(ds.data.length > 100) ds.data.shift(); }); }
        this.chart.update();
    },
    toggleCompare() {
        this.compareMode = !this.compareMode;
        const btn = document.getElementById('btn-compare');
        btn.classList.toggle('bg-yellow-500', this.compareMode);
        btn.classList.toggle('text-black', this.compareMode);
        btn.innerText = this.compareMode ? "🔒 Lock Run (ON)" : "Lock Run (OFF)";
        if(!this.compareMode) changeAlgo();
    },
    reset() {
        const type = document.getElementById('algo-select').value;
        const currentAgentName = AGENT_NAMES[type] || 'Agent';

        if(this.compareMode) {
            // Check if the last dataset has data. If NOT, reuse it.
            // This prevents "Human Run 2, Run 3, Run 4" appearing if you click reset multiple times without playing.
            const datasets = this.chart.data.datasets;
            const lastDataset = datasets[datasets.length - 1];

            if (lastDataset && lastDataset.data.length === 0) {
                // REUSE empty dataset
                lastDataset.label = `${currentAgentName} (Run ${datasets.length})`;
                // keep color
            } else {
                // CREATE NEW dataset only if previous had data
                const colors = ['#bc13fe', '#ffff00', '#00ff00', '#ff0055', '#00f3ff'];
                const newRunNum = datasets.length + 1;
                datasets.push({
                    label: `${currentAgentName} (Run ${newRunNum})`,
                    data: [], 
                    borderColor: colors[(datasets.length) % colors.length], 
                    tension: 0.1, pointRadius: 0, borderWidth: 2 
                });
            }
        } else {
            // Standard Mode: Clear everything
            this.chart.data.labels = [];
            this.chart.data.datasets = [{ label: `${currentAgentName} (Current)`, data: [], borderColor: '#00f3ff', tension: 0.1, pointRadius: 0, borderWidth: 2 }];
        }
        this.chart.update();
    },
    download() {
        const link = document.createElement('a');
        link.download = 'snake-rl-chart.png';
        link.href = document.getElementById('learningChart').toDataURL();
        link.click();
    }
};

// --- 4. GAME ENGINE ---
class SnakeGame {
    constructor() {
        this.board = document.getElementById('game-board');
        this.mode = 'human';
        this.audioCtx = null; this.soundEnabled = false;
        this.running = false;
        chartMgr.init();
        this.setupInput();
        this.episode = 0; this.highScore = 0;
        this.resetRound();
    }

    toggleAudio() {
        if(!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if(this.audioCtx.state === 'suspended') this.audioCtx.resume();
        this.soundEnabled = !this.soundEnabled;
        return this.soundEnabled;
    }
    playSound(type) {
        if(!this.soundEnabled || !this.audioCtx) return;
        const osc = this.audioCtx.createOscillator(); const gain = this.audioCtx.createGain();
        osc.type = type === 'eat' ? 'square' : 'sawtooth';
        osc.frequency.setValueAtTime(type === 'eat' ? 800 : 150, this.audioCtx.currentTime);
        gain.gain.setValueAtTime(0.05, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + 0.1);
        osc.connect(gain); gain.connect(this.audioCtx.destination);
        osc.start(); osc.stop(this.audioCtx.currentTime + 0.1);
    }

    setAgent(type) {
        this.mode = type;
        if(type === 'qlearning') this.agent = new QLearningAgent();
        else if(type === 'sarsa') this.agent = new SarsaAgent();
        else if(type === 'dqn') this.agent = new DQNAgent();
        else this.agent = null; // Human
        this.resetAll();
    }

    resetAll() {
        this.stop();
        this.episode = 0; this.highScore = 0;
        chartMgr.reset(); // Uses the new Smart Reset Logic
        this.resetRound();
        document.getElementById('high-score').innerText = 0;
        document.getElementById('episode-display').innerText = 0;
    }

    resetRound() {
        this.snake = [{x:10, y:10}, {x:10, y:11}, {x:10, y:12}];
        this.food = this.spawnFood();
        this.dir = {x:0, y:-1};
        this.nextDir = {x:0, y:-1}; 
        this.score = 0; this.steps = 0;
        this.updateHUD(); this.render();
    }

    spawnFood() {
        let f; do { f = {x:Math.floor(Math.random()*GRID_SIZE), y:Math.floor(Math.random()*GRID_SIZE)}; }
        while(this.snake.some(s => s.x === f.x && s.y === f.y));
        return f;
    }

    start() {
        if (this.running) return;
        this.running = true;
        
        document.getElementById('overlay').classList.add('hidden');
        const btn = document.getElementById('btn-start');
        btn.innerText = "STOP";
        btn.classList.remove('from-purple-500', 'to-cyan-500');
        btn.classList.add('bg-red-600'); 
        
        this.loop();
    }

    stop() {
        this.running = false;
        
        document.getElementById('overlay').classList.remove('hidden');
        document.getElementById('overlay-title').innerText = this.mode === 'human' ? "GAME OVER" : "PAUSED";
        
        const btn = document.getElementById('btn-start');
        btn.innerText = "START GAME";
        btn.classList.remove('bg-red-600');
        btn.classList.add('from-purple-500', 'to-cyan-500');
    }

    step(action) {
        const moves = [{x:0,y:-1}, {x:0,y:1}, {x:-1,y:0}, {x:1,y:0}];
        const newDir = moves[action];
        if(newDir.x !== -this.dir.x && newDir.y !== -this.dir.y) this.dir = newDir;

        const head = this.snake[0];
        const next = {x: head.x + this.dir.x, y: head.y + this.dir.y};
        let reward = -0.1; let done = false;
        
        const oldDist = Math.abs(head.x - this.food.x) + Math.abs(head.y - this.food.y);
        const newDist = Math.abs(next.x - this.food.x) + Math.abs(next.y - this.food.y);

        if(next.x < 0 || next.x >= GRID_SIZE || next.y < 0 || next.y >= GRID_SIZE || this.snake.some(s => s.x === next.x && s.y === next.y)) {
            reward = -50; done = true;
            if(this.mode === 'human') this.playSound('crash');
        } else {
            this.snake.unshift(next);
            if(next.x === this.food.x && next.y === this.food.y) {
                reward = 50; this.score++; this.food = this.spawnFood(); this.steps = 0;
                if(this.mode === 'human' || this.soundEnabled) this.playSound('eat');
                if(this.score > this.highScore) this.highScore = this.score;
            } else {
                this.snake.pop();
                reward += (newDist < oldDist) ? 0.5 : -0.6;
            }
        }
        
        this.steps++;
        if(this.steps > 500) { done = true; reward = -10; }
        return {reward, done};
    }

    async loop() {
        if(!this.running) return;

        let action = 0;
        
        if(this.mode === 'human') {
            if(this.nextDir.y === -1) action = 0; else if(this.nextDir.y === 1) action = 1;
            else if(this.nextDir.x === -1) action = 2; else if(this.nextDir.x === 1) action = 3;
            
            const res = this.step(action);
            if(res.done) { 
                chartMgr.addPoint(this.episode, this.score);
                this.episode++;
                this.stop(); 
                this.resetRound(); 
                return; 
            }
        } else {
            const state = this.agent.getStateVector(this);
            action = this.agent.getAction(state);
            const res = this.step(action);
            const nextState = this.agent.getStateVector(this);
            
            if(this.mode === 'qlearning') this.agent.learn(state, action, res.reward, nextState, res.done);
            else if(this.mode === 'sarsa') {
                const nextAction = this.agent.getAction(nextState);
                this.agent.learn(state, action, res.reward, nextState, res.done, nextAction);
            } else if(this.mode === 'dqn') {
                this.agent.remember(state, action, res.reward, nextState, res.done);
                await this.agent.replay();
            }

            if(res.done) {
                chartMgr.addPoint(this.episode, this.score);
                this.episode++;
                this.resetRound();
            }
        }

        this.render();
        this.updateHUD();
        
        // --- CALIBRATED SPEED CONTROL ---
        let speedDelay = 100;
        const speedMode = document.getElementById('btn-speed').dataset.speed;
        
        // TWEAK: Fast Mode set to 70ms (slower than 50ms, faster than 100ms)
        if(speedMode === 'fast') speedDelay = 70; 
        else if(speedMode === 'hyper') speedDelay = 0; 

        // Cap speed for human in hyper mode
        if(this.mode === 'human' && speedDelay === 0) speedDelay = 30;

        if(speedDelay === 0) requestAnimationFrame(() => this.loop());
        else setTimeout(() => this.loop(), speedDelay);
    }

    setupInput() {
        window.addEventListener('keydown', e => {
            if(e.key === ' ' && !this.running) { this.start(); return; }
            if(e.key === ' ' && this.running) { this.stop(); return; }
            if(this.mode !== 'human') return;
            switch(e.key) {
                case 'ArrowUp': if(this.dir.y !== 1) this.nextDir = {x:0, y:-1}; break;
                case 'ArrowDown': if(this.dir.y !== -1) this.nextDir = {x:0, y:1}; break;
                case 'ArrowLeft': if(this.dir.x !== 1) this.nextDir = {x:-1, y:0}; break;
                case 'ArrowRight': if(this.dir.x !== -1) this.nextDir = {x:1, y:0}; break;
            }
        });
    }

    render() {
        if(this.mode !== 'human') {
             const speedMode = document.getElementById('btn-speed').dataset.speed;
             if(speedMode === 'fast' && this.steps % 5 !== 0) return;
             if(speedMode === 'hyper' && this.steps % 50 !== 0) return;
        }

        this.board.innerHTML = '';
        const f = document.createElement('div');
        f.className = 'food-item absolute';
        f.style.left = this.food.x*CELL_SIZE+'px'; f.style.top = this.food.y*CELL_SIZE+'px';
        f.style.width = CELL_SIZE+'px'; f.style.height = CELL_SIZE+'px';
        f.innerHTML = '<div class="food-glow"></div>';
        this.board.appendChild(f);

        this.snake.forEach((s, i) => {
            const el = document.createElement('div');
            el.className = `snake-segment ${i===0?'snake-head':''}`;
            el.style.left = s.x*CELL_SIZE+'px'; el.style.top = s.y*CELL_SIZE+'px';
            el.style.width = CELL_SIZE+'px'; el.style.height = CELL_SIZE+'px';
            el.style.zIndex = 100-i;
            el.innerHTML = '<div class="snake-cube"></div>';
            this.board.appendChild(el);
        });
    }

    updateHUD() {
        document.getElementById('score-display').innerText = this.score;
        document.getElementById('episode-display').innerText = this.episode;
        document.getElementById('high-score').innerText = this.highScore;
        if(this.agent) document.getElementById('epsilon-display').innerText = Math.round(this.agent.epsilon*100) + '%';
        else document.getElementById('epsilon-display').innerText = "--";
    }
}

// --- 5. INITIALIZATION ---
const game = new SnakeGame();

function toggleGame() { 
    if(game.running) game.stop(); 
    else game.start(); 
}

function toggleSpeed() {
    const btn = document.getElementById('btn-speed');
    const currentMode = btn.dataset.speed;
    let newMode, newText;

    if (currentMode === 'normal') {
        newMode = 'fast'; newText = "⚡ Speed: Fast";
    } else if (currentMode === 'fast') {
        newMode = 'hyper'; newText = "🚀 Speed: HYPER";
    } else {
        newMode = 'normal'; newText = "⚡ Speed: Normal";
    }

    btn.dataset.speed = newMode;
    btn.innerText = newText;
}

function toggleAudio() {
    const isOn = game.toggleAudio();
    const btn = document.getElementById('btn-audio');
    btn.innerText = isOn ? "🔊 Sound: ON" : "🔈 Sound: OFF";
    btn.classList.toggle('text-green-400', isOn);
}

function changeAlgo() {
    const type = document.getElementById('algo-select').value;
    game.setAgent(type);
    document.getElementById('status-text').innerText = AGENT_NAMES[type].toUpperCase();
    document.getElementById('overlay-title').innerText = "READY";
    
    if(!chartMgr.compareMode && chartMgr.chart && chartMgr.chart.data.datasets.length > 0) {
         chartMgr.chart.data.datasets[0].label = AGENT_NAMES[type] + ' (Current)';
         chartMgr.chart.update();
    }
}

function resetAgent() {
    game.stop();
    changeAlgo(); 
}

window.onload = function() {
    document.getElementById('algo-select').value = 'human';
    changeAlgo();
};
