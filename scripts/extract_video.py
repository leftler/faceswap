import cv2
import os
from scripts.extract import ExtractTrainingData
from lib.utils import get_video_paths, get_folder
from plugins.PluginLoader import PluginLoader
from moviepy.editor import VideoFileClip
from imageio.plugins import ffmpeg


class ExtractTrainingDataVideo(ExtractTrainingData):
    frames = 0
    currentfile = ""

    def create_parser(self, subparser, command, description):
        self.parser = subparser.add_parser(
            command,
            help="Extract the faces from a video source.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )

    def add_optional_arguments(self, parser):
        parser.add_argument('-D', '--detector',
                            type=str,
                            choices=("hog", "cnn"), # case sensitive because this is used to load a plugin.
                            default="hog",
                            help="Detector to use. 'cnn' detects much more angles but will be much more resource intensive and may fail on large files.")

        parser.add_argument('-f', '--filter',
                            type=str,
                            dest="filter",
                            default="filter.jpg",
                            help="Reference image for the person you want to process. Should be a front portrait"
                            )

        return parser

    def start(self):
        self.output_dir = get_folder(self.arguments.output_dir)
        try:
            self.input_dir = get_video_paths(self.arguments.input_dir)
        except:
            print('Input directory not found. Please ensure it exists.')
            exit(1)

        self.filter = self.load_filter()
        self.process()
        self.finalize()

    def process(self):        
        ffmpeg.download()
        extractor_name = "Align" # TODO Pass as argument
        self.extractor = PluginLoader.get_extractor(extractor_name)()

        try:
            for filename in self.read_directory():
                print('Loading %s' % (filename))
                self.frames = 0
                self.currentfile = filename
                input = VideoFileClip(filename)
                clip = input.fl_image(self.processFrame)#.subclip(11, 13) #NOTE: this function expects color images!!                
                clip.write_videofile("_.mp4", audio=False)
                os.remove("_.mp4")
        finally:
            self.write_alignments()
        

    def processFrame(self, image):
        self.frames += 1
        fname = os.path.basename(self.currentfile) + "_f" + str(self.frames)
        try:
            self.faces_detected[fname] = self.handleImage(image, fname)
        except Exception as e:
            print('Failed to extract from frame: {}. Reason: {}'.format(fname, e))
        return image
            
    
