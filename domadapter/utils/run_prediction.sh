PYTHON_FILE=${PROJECT_ROOT}/"domadapter/utils/compare_adapters.py"

# info about array
# key -> adapter folder location
# value -> name of its corresponding output CSV file

declare -A array
array["./task_adapter"]="domain_adapter.csv"
array["./231414"]="task_adapter.csv"

for i in "${!array[@]}"
do
     python ${PYTHON_FILE} \
        --adapter ${i} \
        --output  ${array[$i]} \
        --dataset "/home/bhavitvya/work/domadapter/data/mnli/fiction_government/test_target.csv"
done