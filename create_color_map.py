import cv2
import numpy as np
import logging
import enlighten
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class ColorMapMaker:
    def __init__(
        self,
        video_file_path: str,
        image_size: int,
        color_section_size: int,
        flip_to_vertical: bool,
        sampling_interval: int,
    ):
        """
        :param video_file_path: Path to input video file of which color map will be created. Format support is determined by OpenCV support.
        :param image_size: Size (in pixels) of one of the images axis - height for default image, width for flipped (vertical).
                          Other axis is defined by video length, sampling interval and color section size.
        :param color_section_size: Size in pixels (height or width depending if image is horizontal or vertical) of mean color segment in the result image.
        :param flip_to_vertical: Decides whether image is presented vertically or horizontally
        :param sampling_interval: Number of frames between which mean color values is sampled (default 24 is one sample per second for most movies)
        """
        self._video = cv2.VideoCapture(video_file_path)
        self._image_size = image_size
        self._color_section_size = color_section_size
        self._flip_to_vertical = flip_to_vertical
        logger.debug(
            f"Received following attributes: \n"
            + f" -video path = {video_file_path}\n"
            + f" -image_size = {image_size}\n"
            + f" -color_section_size = {color_section_size}\n"
            + f" -flip_to_vertical = {flip_to_vertical}\n"
            + f" -sampling_interval = {sampling_interval}"
        )
        self._total_video_frames = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._total_video_frames == 0:
            logger.error('Video was not loaded correctly, exiting.')
            exit(1)
        number_of_color_sections = self._total_video_frames // sampling_interval
        logger.debug(
            f"Detected {self._total_video_frames} frames in loaded video, sampling {number_of_color_sections} colors"
        )

        self.result_image = np.zeros(
            [self._image_size, self._color_section_size * number_of_color_sections, 3]
        )

    def process_image(self) -> None:
        """
        Processes loaded file and creates color map in memory. 
        """
        logger.info("Started processing loaded image.")
        manager = enlighten.get_manager()
        progress_bar = manager.counter(
            total=self._total_video_frames, desc="Processed frames", unit="Frames"
        )
        processed_frame_n = 0
        processed_color_section_n = 0
        for processed_frame_n in range(self._total_video_frames):
            retval, image = self._video.read()  # Grab a frame from loaded video
            if retval:
                if processed_frame_n % 24 == 0:
                    mean_color = np.mean(image, axis=(0, 1))
                    mean_image = (
                        np.ones([self._image_size, self._color_section_size, 3])
                        * mean_color
                    )
                    # Assign image with mean color as a section in result image
                    self.result_image[
                        :,
                        (processed_color_section_n * self._color_section_size) : (
                            processed_color_section_n * self._color_section_size
                            + self._color_section_size
                        ),
                        :,
                    ] = mean_image
                    processed_color_section_n += 1
            else:
                break
            progress_bar.update()
        if self._flip_to_vertical:
            self.result_image = cv2.rotate(self.result_image, cv2.ROTATE_90_CLOCKWISE)
    
    def save_result_to_file(self, output_file_path: str) -> None:
        retval = cv2.imwrite(output_file_path, self.result_image)
        if retval:
            logger.info(f'Succcesfully saved result image to {output_file_path}')
        else:
            logger.error('Saving result image failed.')

    def cleanup(self) -> None:
        self._video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Color Map Properties")
    parser.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="Path to input video file of which color map will be created. Format support is determined by OpenCV support.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Path defining where output image will be saved.",
    )
    parser.add_argument(
        "-s",
        "--size",
        required=False,
        default=3000,
        type=int,
        help="Size (in pixels) of one of the images axis - height for default image, width for flipped (vertical). Other axis is defined by video length, sampling interval and color section size. Default is 3000",
    )
    parser.add_argument(
        "-css",
        "--color-section-size",
        default=1,
        type=int,
        required=False,
        help="Size in pixels (height or width depending if image is horizontal or vertical) of mean color segment in the result image. Default is 1.",
    )
    parser.add_argument(
        "-f",
        "--flip",
        required=False,
        default=False,
        action='store_true',
        help="Decides whether image is presented vertically or horizontally. By default image is horizontal, setting this makes it vertical.",
    )
    parser.add_argument(
        "-si",
        "--sampling-interval",
        default=24,
        type=int,
        required=False,
        help="Number of frames between which mean color values is sampled. Default is 24, one sample per second for most movies.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        action="store_true",
        help="Determines how much information is included in logs",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    cmm = ColorMapMaker(
        video_file_path=args.input_path,
        image_size=args.size,
        color_section_size=args.color_section_size,
        flip_to_vertical=args.flip,
        sampling_interval=args.sampling_interval,
    )
    cmm.process_image()
    cmm.save_result_to_file(args.output_path)
    cmm.cleanup()
