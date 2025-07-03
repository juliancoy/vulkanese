# Rename all *.txt to *.text
for file in *.c; do 
    mv -- "$file" "${file%.c}.comp.template"
done
