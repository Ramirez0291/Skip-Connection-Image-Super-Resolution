# model and training
MODEL_NAME = 'mod2plusY'
BATCH_SIZE = 4
INPUT_SIZE = 120  # the image size input to the network
INPUT_SIZE_RUN = 120 # the image size input to the network when you use to rebulid your image
SCALE_FACTOR = 2
LABEL_SIZE = SCALE_FACTOR * INPUT_SIZE  # the high resolution image size used as label
LABEL_SIZE_RUN = SCALE_FACTOR * INPUT_SIZE_RUN
NUM_CHENNELS = 3

# data path and log path
#INPUT_IMAGE_PATH = './test/Input_image'
INPUT_IMAGE_PATH = './test/Original'
ORIGINAL_TEST_PATH = './test/Original'
OUTPUT_IMAGE_PATH = './test/Output_image'
TEST_IMAGE_PATH = './test/set14'
ORIGINAL_IMAGES_PATH = './Image2X/data/original'
TRAINING_DATA_PATH = './Image2X/data/test'
VALIDATION_DATA_PATH = './Image2X/data/valid'
TESTING_DATA_PATH = './test/set14'
INFERENCES_SAVE_PATH = './Image2X/' + MODEL_NAME + '/inferences'
TRAINING_SUMMARY_PATH = './Image2X/' + MODEL_NAME + '/training_summary'
CHECKPOINTS_PATH = './Image2X/' + MODEL_NAME + '/checkpoints'
KODAK_TEST_SET ='./test/KODAK'
MAX_CKPT_TO_KEEP = 50  # max checkpoint files to keep

# patch generation
PATCH_SIZE = 80  # must be even, the image size croped from original images
PATCH_GEN_STRIDE = 32  # maybe used by data generation
PATCH_RAN_GEN_RATIO = 2  # the number of random generated patches is max(img.height, img.width) // PATCH_RAN_GEN_RATIO

# data queue
MIN_QUEUE_EXAMPLES = 12
NUM_PROCESS_THREADS = 3
NUM_TRAINING_STEPS = 5000000
NUM_TESTING_STEPS = 10

# data argumentation
MAX_RANDOM_BRIGHTNESS = 0.2
RANDOM_CONTRAST_RANGE = [0.8, 1.2]
GAUSSIAN_NOISE_STD = 0  # [0...1] (float)
JPEG_NOISE_LEVEL = 0  # [0...4] (int)
