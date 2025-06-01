from frechet_video_distance import frechet_video_distance as fvd

def calculate_fvd_score(real_videos, generated_videos):
    # real_videos and generated_videos are tensors of 
    # [NUMBER_OF_VIDEOS, VIDEO_LENGTH, FRAME_WIDTH, FRAME_HEIGHT, 3] with values in 0-255
    result = fvd.calculate_fvd(
        fvd.create_id3_embedding(fvd.preprocess(real_videos, (224, 224))),
        fvd.create_id3_embedding(fvd.preprocess(generated_videos, (224, 224))))
    return result

def calculate_is_score(stylized_frame):
    return 0