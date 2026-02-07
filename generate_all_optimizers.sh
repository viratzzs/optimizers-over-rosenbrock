
export NUM_STEPS=7000

generate_video() {
    local opt_name=$1
    local lr=$2
    local output_file=$(echo "$opt_name" | tr '[:upper:]' '[:lower:]').mp4
    
    echo "Generating $opt_name (lr=$lr)..."
    export OPTIMIZER_NAME="$opt_name"
    export LEARNING_RATE="$lr"
    manim -qm optimizers/utils/visualizer2d.py Optimizer2D -o "$output_file"
}

optimizers_001=("Adam" "AdaMax" "Nadam" "RMSprop" "Adadelta" "Adagrad")
for opt in "${optimizers_001[@]}"; do
    generate_video "$opt" "0.001"
done

optimizers_0001=("Momentum" "Nesterov")
for opt in "${optimizers_0001[@]}"; do
    generate_video "$opt" "0.0001"
done

echo "All visualizations generated!"
