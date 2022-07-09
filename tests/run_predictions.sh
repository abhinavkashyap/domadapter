PYTHON_FILE=${PROJECT_ROOT}/"tests/test_domadapter.py"
# info about array
# key -> adapter folder location
# value -> name of its corresponding output CSV file

# DOMAIN_ADAPTER_WANDB_id=(1q46nbuw 1no49m56 1ta8oiqr 2j6ff13c 1pb53tgd 335aad8u 2vybdw6j x5a2p3ly dm9zvye6 2hfbgaqv 2t7p8mb2 56g93ojx 8npxjztj 2dl5vh06 2g7yr9oj 1xirez89 189bu86l 1tcqy1ga 215rrjpn)
# DOMAINS=("apparel_baby" apparel_books apparel_camera_photo apparel_MR baby_apparel baby_books baby_camera_photo baby_MR books_apparel books_baby books_camera_photo books_MR camera_photo_apparel camera_photo_books camera_photo_MR MR_apparel MR_baby MR_books MR_camera_photo)
DOMAIN_ADAPTER_WANDB_id=(39jksxe5 2ugczeba 3hmqm3nb c0ljw8gt 33ek7144 20ivpyqk 2tnf4e40 6yc9xrjv si7f4v73 19fcbumj 1bb5j40t 1u4okcep 13ua6c9c 1b8eould j0harhpu 28feoj08 78zq1ow0 1cgaqh7i)
DOMAINS=("fiction_government" fiction_telephone fiction_travel slate_fiction slate_government slate_telephone government_fiction government_slate government_telephone government_travel telephone_fiction telephone_slate telephone_government telephone_travel travel_fiction travel_slate travel_government travel_telephone)

for ((i=0; i<=17; i++)); do
    python ${PYTHON_FILE} \
        --domain-adapter "/ssd1/abhinav/domadapter/experiments/${DOMAINS[i]}/domain_adapter/${DOMAIN_ADAPTER_WANDB_id[i]}/checkpoints" \
        --task-adapter "/home/bhavitvya/domadapter/experiments/slate_travel/ablations_task_adapter_mkmmd/1f11mm1a/checkpoints" \
        --data-module "mnli" \
        --dataset "/home/bhavitvya/domadapter/data/mnli/${DOMAINS[i]}/test_target.csv"
    COUNTER=$[$COUNTER +1]
done