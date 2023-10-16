from argparse import ArgumentParser

class InferOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.register_options()

    def register_options(self):
        self.parser.add_argument("--device", help="The device to perform inference on. Must be either \"cpu\" or \"cuda\"")
        self.parser.add_argument("--config_path", help="Full path to config file. Default is config/model_config.py", 
                                    default="config/model_config.py", required=True)
        self.parser.add_argument("--checkpoint_path", help="Full path to the checkpoint file. Default is checkpoints/gen_420000.pt", 
                                    default="checkpoints/gen_420000.pt", required=True)
        self.parser.add_argument("--input_path", help="Path to the input image.", required=True)
        self.parser.add_argument("--output_path", help="Desired path to output folder to save the images to.", required=True)
        self.parser.add_argument("--mode", help="The mode for translation \"l\" for latent-guided, \"r\" for reference-guided.", required=True)
        self.parser.add_argument("--tag", help="The desired tag to be edited. For the ordering of the tags, check config/model_config.py", type=int)
        
        # Latent-guided options
        self.parser.add_argument("--attribute", help="The target attribute for the translation. \
                                                    For the ordering of the attributes, check config/model_config.py.", type=int)
        self.parser.add_argument("--z", help="Scalar value describing the strength of the edit. The value should be within the interval [0,1]", \
                                 type=float)

        # Reference-guided options
        self.parser.add_argument("--reference_path", help="Path to the reference image.")

    def parse_args(self):
        opts = self.parser.parse_args()
        return opts