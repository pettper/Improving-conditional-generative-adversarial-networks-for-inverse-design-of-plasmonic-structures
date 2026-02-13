shapes=("diamond" "prism" "dimer_export" "ellipsoid")

for s in "${shapes[@]}"; do
    train=$(ls -1 anisotropic_au_structures_train_val_test/training/featherfiles/ | grep -c "$s")
    val=$(ls -1 anisotropic_au_structures_train_val_test/validation/featherfiles/ | grep -c "$s")
    test=$(ls -1 anisotropic_au_structures_train_val_test/test/featherfiles/ | grep -c "$s")
    
    total=$((train + val + test))
    
    echo "$s: $total (Train: $train, Val: $val, Test: $test)"
done

echo "Training set: " $(ls -1 anisotropic_au_structures_train_val_test/training/featherfiles/ | wc -l)
echo "Validation set: "$(ls -1 anisotropic_au_structures_train_val_test/validation/featherfiles/ | wc -l)
echo "Test: " $(ls -1 anisotropic_au_structures_train_val_test/test/featherfiles/ | wc -l)