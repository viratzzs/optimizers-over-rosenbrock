# Array of optimizers
optimizers=("Nesterov" "Adagrad" "RMSprop" "Adadelta" "AdaMax" "Nadam")

for opt in "${optimizers[@]}"; do
    echo "Generating visualization for $opt..."
    
    # Update visualizer to use current optimizer
    sed -i "s/sim = Simulator([^,]*,/sim = Simulator($opt(lr=0.0001),/" optimizers/utils/visualizer.py
    
    # Render with manim  
    D:/projects/from-scratch/.venv/Scripts/python.exe -m manim -pql optimizers/utils/visualizer.py Optimizer2D -o ${opt,,}.mp4
    
    # Move output to root
    if [ -f "media/videos/visualizer/480p15/$(echo $opt | tr '[:upper:]' '[:lower:]').mp4" ]; then
        mv "media/videos/visualizer/480p15/$(echo $opt | tr '[:upper:]' '[:lower:]').mp4" .
    fi
    
    echo "Completed $opt"
    echo "---"
done

echo "All optimizer visualizations generated!"
