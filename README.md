# AI Race – Genetic Algorithm + Neural Network Self-Driving Boxes

This project simulates small “car/box” agents learning to drive through a curvy S-shaped track.  
Each agent has:

- A simple neural network (5 sensors + speed input → steering + acceleration output)
- Genetic algorithm evolution (elitism, crossover, mutation)
- Checkpoint-based progress system
- Finish-line detection and bonus scoring
- Optional visual debugging (sensors & checkpoints visible/invisible)

The agents start with random behaviour and gradually learn to complete the entire track.

---

## 🚗 Features

### ✔ Pygame visual simulation  
The environment renders the map (`track.png`) and all 40 cars in real time.

### ✔ Neural network per car  
Each car has a small custom NN:
- 6 inputs
- 8 hidden neurons
- 2 outputs (steering, accelerate)

### ✔ Genetic Algorithm  
Every generation:
- **Finishers are always preserved**  
- **Top elites are copied unchanged**
- Children are produced by **crossover + mutation**

### ✔ Checkpoints  
Cars must pass checkpoints in order.  
This prevents U-turn tricks and ensures real progress.

### ✔ Manual skip  
Press **SPACE** to kill all cars and move to next generation instantly.

---

## 🖼 Project Structure

ai_race/
│── main.py
│── track.png
│── README.md
│── .gitignore
└── (optional later: assets/, docs/)


## ▶ How to Run

### 1. Install requirements
pip install pygame

### 2. Run the simulation
python main.py

### 3. Controls

| Key | Action |
|-----|--------|
| SPACE | Force end generation (skip to next) |
| ESC (close window) | Exit program |

---

## 🧠 How the Learning Works

Each car has:
- **Sensors** that detect distance to walls
- A **neural network** that outputs steering & acceleration
- A **fitness score** based on:
  - Number of checkpoints reached
  - Distance to next checkpoint
  - Survival time
  - Big finish bonus

Evolution happens using:
- **Elitism (best cars cloned)**
- **Crossover between best parents**
- **Mutation (controlled randomness)**

Over many generations, cars learn smooth driving lines and reach the finish consistently.

---

## 🎥 Add a GIF / Screenshot (optional)

You can add a GIF or PNG like:

Use ScreenToGif or OBS to capture.

---

## 📌 Future Improvements

- Save & load best-performing neural networks
- Add obstacles / multi-lane tracks
- GUI controls for mutation rate, speed, population size
- 3D version using Unity or Blender

---

## 📜 License
This project is open for learning and experimentation.