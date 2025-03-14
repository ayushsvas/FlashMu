config = {
    "test":{
        "dfc": False,
        "unet": False,
        "device": 'cuda'
    },
    "train":{
        "dfc": True,
        "unet": False,
        "device": 'cuda',
        "data_parallel": False
    },

    "dfc":{
        "fourier_part": {
            "n_modes" : 64, # 256, 128, 96, 48, 64, 32, 384, 768
            "hidden_channels" : 128,
            "in_channels" : 3, 
            "out_channels" : 1,
            "lifting_channels" : 128,
            "projection_channels" : 128,
            "n_layers" : 3,
            "separable_fourier_layers":  [False, *[False]*3],#[False, *[True]*3],#[False, *[True]*3], # multiply with n_layers
            "dilate_fourier_kernel_fac" : 1, #1, 2, 4
            "fourier_interpolation": False,
            "skip" : 'conv3',#'linear'
            "factorization" : 'dense',#'cp',#'tucker', # 'dense'
            "implementation": 'factorized',
            "rank" : 1, #1/20000, # 0.51, # 0.34, 1.0
            "mem_checkpoint": False,
            "bias": True, # False
            "fourier_block_precision": "full",
            "batch_norm": False,
            },

        "dilated_cnn_part": {
            "kernel_size" : 5, # 5, # None
            "dilations": [1,3,9], #[1,3,9], # [None]*3
            "padding": True, # True,# None
            },

        "data":{
            "IMG_DIR" : "/mnt/lustre-grete/projects/eckerlab/holography/4096x4096_data/1536x1536_crops/synthetic_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy",#"/mnt/lustre-grete/projects/eckerlab/holography/deploy_train_data/synthetic_flight06_flight12_holograms.npy", #"/project.lmp/cloudkite-proc/2048x2048_data/synthetic_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy", #"/project.lmp/cloudkite-proc/4096x4096_data/512x512_crops/holo_crops_864000x128x128_4DS.npy",#"/project.lmp/cloudkite-proc/2048x2048_data/synthetic_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy", #"/project.lmp/cloudkite-proc/4096x4096_data/1024x1024_crops/holo_crops_216000x256x256_4DS.npy",#"/project.lmp/cloudkite-proc/4096x4096_data/512x512_crops/holo_crops_864000x128x128_4DS.npy",
            "MASK_DIR": "/mnt/lustre-grete/projects/eckerlab/holography/4096x4096_data/1536x1536_crops/predicted_synthetic_weighted_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy",#"/mnt/lustre-grete/projects/eckerlab/holography/deploy_train_data/synthetic_flight06_flight12_weighted_holograms.npy", #None, #"/project.lmp/cloudkite-proc/4096x4096_data/512x512_crops/weighted_holo_crops_864000x128x128_4DS.npy", #"/project.lmp/cloudkite-proc/2048x2048_data/predicted_synthetic_weighted_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy", #"/project.lmp/cloudkite-proc/2048x2048_data/synthetic_weighted_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy",#"/project.lmp/cloudkite-proc/4096x4096_data/1024x1024_crops/weighted_holo_crops_216000x256x256_4DS.npy",
            "SAVE_FOLDER" : "/user/apaliwa/u12018/data/saved_predictions_0",
            "LOAD_CHECKPOINT_DIR" : ""
            },

        "training": {
            "IMG_SIZE": 384,#256, #128, 384, 768
            "NUM_SAMPLES": 121000,# 216000,54000,108000, 63000
            "LEARNING_RATE" : 1e-4*8/10,
            "DEVICE" : 'cuda', 
            "VAL_FRAC" : 0.001,
            "SPLIT_SEED" : 42,
            "BATCH_SIZE" : 8, #10,  #64
            "NUM_EPOCHS" : 121,
            "NUM_WORKERS" : 2,
            "PIN_MEMORY" : True,
            "LOAD_MODEL" : False,
            "AUGMENT" : True,
            "BACKGROUND_AUGMENT" : True, # False
            "BACKGROUND_DIR":"/mnt/lustre-grete/projects/eckerlab/holography/4096x4096_data/1536x1536_crops/sequential_overlapping_rotated_1536x1536_2213BackgroundImages.npy",#"/mnt/lustre-grete/projects/eckerlab/holography/deploy_train_data/sequential_overlapping_rotated_1536x1536_2213BackgroundImages_2DS.npy", #"/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_1536x1536_2213BackgroundImages.npy", #"/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_128x128_3035BackgroundImages.npy", # "/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_1536x1536_2213BackgroundImages.npy",#None,# "/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_1024x1024_304BackgroundImages.npy", #"/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_128x128_3035BackgroundImages.npy",
            "SCHEDULER_COUNT" : 2,
            "LR_DECAY_FAC" : 0.5,
            "EARLY_STOPPING_COUNT" : 5,
            "MMAP_MODE": 'r',
            "mean": [124.52455139160156], 
            "std": [15.454506874084473],
            },

        "evaluation":{
            "ds_factor": 4,
            "HOLO_SIZE": 5120,
            "crop_size": 1536,#1024, 512, 1536
            "step_size": 1536, #512
            "dist_to_cut_from_edge": 256,
            "mean": [124.85907745361328], 
            "std": [18.08700180053711],
            "optimal_batch_size": [64, 48, 32, 24, 16, 8, 4, 2, 1],  
            }
        },

    "unet":{
        "build": {
        "name": 'pi_unet',
        "in_channels": 2, # 1,2
        "out_channels": 3, 
        "in_size": 384, #768, #384, #256,128, 384
        "latent_lift_proj_dim": 64, #512, 64
        "n_layers":6,
        "n_res": 2,
        "activations": 'lrelu',    
        "loss": 'shao+pi_loss'
    },
        "data": {
            "IMG_DIR" : "/mnt/lustre-grete/projects/eckerlab/holography/4096x4096_data/1536x1536_crops/synthetic_holograms_NEW384x384_121k_from_4096x4096_data.npy",# /mnt/lustre-grete/projects/eckerlab/holography/4096x4096_data/1536x1536_crops/synthetic_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy",#"/mnt/lustre-grete/projects/eckerlab/holography/deploy_train_data/synthetic_flight06_flight12_holograms.npy", #"/project.lmp/cloudkite-proc/2048x2048_data/synthetic_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy", #"/project.lmp/cloudkite-proc/4096x4096_data/512x512_crops/holo_crops_864000x128x128_4DS.npy",#"/project.lmp/cloudkite-proc/2048x2048_data/synthetic_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy", #"/project.lmp/cloudkite-proc/4096x4096_data/1024x1024_crops/holo_crops_216000x256x256_4DS.npy",#"/project.lmp/cloudkite-proc/4096x4096_data/512x512_crops/holo_crops_864000x128x128_4DS.npy",
            "WIMG_DIR": "/mnt/lustre-grete/projects/eckerlab/holography/4096x4096_data/1536x1536_crops/predicted_synthetic_weighted_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy",#"/mnt/lustre-grete/projects/eckerlab/holography/deploy_train_data/synthetic_flight06_flight12_weighted_holograms.npy", #None, #"/project.lmp/cloudkite-proc/4096x4096_data/512x512_crops/weighted_holo_crops_864000x128x128_4DS.npy", #"/project.lmp/cloudkite-proc/2048x2048_data/predicted_synthetic_weighted_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy", #"/project.lmp/cloudkite-proc/2048x2048_data/synthetic_weighted_holograms_121kx384x384_from_4096x4096atQuarterResolution.npy",#"/project.lmp/cloudkite-proc/4096x4096_data/1024x1024_crops/weighted_holo_crops_216000x256x256_4DS.npy",
            "MASK_DIR" : "/mnt/lustre-grete/projects/eckerlab/holography/4096x4096_data/synthetic_pparas_4096x4096_1DS_13p5kSamples4096_1536_1280_xypx_binned.pkl",#"/mnt/lustre-grete/projects/eckerlab/holography/deploy_train_data/synthetic_flight06_flight12_pparas_new_xypx_binned.pkl", #"/project.lmp/cloudkite-proc/4096x4096_data/synthetic_pparas_4096x4096_1DS_13p5kSamples.pkl", #"/project.lmp/cloudkite-proc/4096x4096_data/synthetic_pparas_4096x4096_1DS_13p5kSamples_xypx_binned.pkl",#"/project.lmp/cloudkite-proc/4096x4096_data/synthetic_pparas_4096x4096_1DS_13p5kSamples.pkl",#"/project.lmp/cloudkite-proc/4096x4096_data/512x512_crops/weighted_holo_crops_864000x128x128_4DS.npy", 
            "SAVE_FOLDER" : "/user/apaliwa/u12018/data/saved_predictions_0",
            "LOAD_CHECKPOINT_DIR" : ""
        },
        "training": {
            "IMG_SIZE": 384,#768, #256, #128, 384
            "NUM_SAMPLES": 121000,#,106176-32400, #121000, #32400, #106176, #864000, #864000,#121000, #216000,# 216000,54000,108000 
            "LEARNING_RATE" : 5.0e-5*20/32,
            "DEVICE" : 'cuda', 
            "VAL_FRAC" : 0.001,
            "SPLIT_SEED" : 42,# 420, 421, 422, 
            "BATCH_SIZE" : 20,  
            "NUM_EPOCHS" : 102,
            "NUM_WORKERS" : 2,
            "PIN_MEMORY" : True,
            "LOAD_MODEL" : False,
            "AUGMENT" : True,
            "BACKGROUND_AUGMENT" : True, # False, True
            "BACKGROUND_DIR":"/mnt/lustre-grete/projects/eckerlab/holography/4096x4096_data/1536x1536_crops/sequential_overlapping_rotated_1536x1536_2213BackgroundImages.npy",#"/mnt/lustre-grete/projects/eckerlab/holography/deploy_train_data/sequential_overlapping_rotated_1536x1536_2213BackgroundImages_2DS.npy", #"/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_1536x1536_2213BackgroundImages.npy", #"/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_128x128_3035BackgroundImages.npy", # "/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_1536x1536_2213BackgroundImages.npy",#None,# "/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_1024x1024_304BackgroundImages.npy", #"/data.lmp/apaliwal/normalized_bkg_crops_from_BirtesSetup/sequential_overlapping_rotated_128x128_3035BackgroundImages.npy",
            "SCHEDULER_COUNT" : 8,
            "LR_DECAY_FAC" : 0.7,
            "EARLY_STOPPING_COUNT" : 20,
            "MMAP_MODE": None,
            "SAVE_FREQ": 3,
            "ALPHA": 0.0001,
            "BETA": 0.0005,
            "mean":  [124.52455139160156, 0],#[126.93829, 0],
            "std": [15.454506874084473, 1]#[13.90818, 1],
            },

        "evaluation":{
            "gkern_size" : 7, # size of the gaussian blob, controls the strictness of the hitbox
            # Evaluate for preicison and recall
            "cutoff_range": 100, # will check 100 values between 0 and 1
            "min_distance": 2,# min distance b/w consecutive peaks in the prediction, can controle this
            "error_calc": False,
            "max_r" : 75, # max particle size in the gt, µm
            "min_r" : 6.0, # min particle size in the gt 
            "min_z" : 5.0, # mm
            "max_z": 200.0,
            "ez_allowed" : 10, # if error in z greater than 10mm, count the prediction as fasle postive
            "mean":  [126.93829, 0],
            "std": [13.90818, 1],
        }
    },

    }

config["dfc"]["architecture"] = '_f_'+'_'.join(map(str, [config["dfc"]["fourier_part"][key] for key in config["dfc"]["fourier_part"]]))+"_c_"+'_'.join(map(str, [config["dfc"]["dilated_cnn_part"][key] for key in config["dfc"]["dilated_cnn_part"]]))

config["dfc"]["data"]["CHECKPOINT_DIR"] = "/mnt/lustre-grete/projects/eckerlab/holography/checkpoints/ablations/ablation_checkpoint_dfc_"+\
                                    str(config["dfc"]["training"]["NUM_SAMPLES"])+"x"+str(config["dfc"]["training"]["IMG_SIZE"])+"x"+\
                                    str(config["dfc"]["training"]["IMG_SIZE"])+"_"+\
                                    config["dfc"]["architecture"]+"_bg"+str(config["dfc"]["training"]["BACKGROUND_AUGMENT"])+".pth.tar"

config["dfc"]["data"]["LOAD_CHECKPOINT_DIR"] = config["dfc"]["data"]["CHECKPOINT_DIR"] # set this if you have your own weights



config["unet"]["architecture"] = '_'.join(map(str, [config["unet"]["build"][key] for key in config["unet"]["build"]]))

config["unet"]["data"]["CHECKPOINT_DIR"] = "/mnt/lustre-grete/projects/eckerlab/holography/checkpoints/checkpoint_unet_"+\
                                    str(config["unet"]["training"]["NUM_SAMPLES"])+"x"+str(config["unet"]["training"]["IMG_SIZE"])+"x"+\
                                    str(config["unet"]["training"]["IMG_SIZE"])+"_"+\
                                    config["unet"]["architecture"]+".pth.tar"





        