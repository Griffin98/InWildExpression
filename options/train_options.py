from argparse import ArgumentParser


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', default="InWildExpression", type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--dataset_dir', default='data/ffhq/', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--expression_dim', default=50, type=int,
                                 help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--output_size', default=512, type=int,
                                 help='Output size of generator')

        self.parser.add_argument('--batch_size', default=4, type=int,
                                 help='Batch size for training')
        self.parser.add_argument('--workers', default=16, type=int,
                                 help='Number of train dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.002, type=float,
                                 help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='adam', type=str,
                                 help='Which optimizer to use')

        # StyleGAN options
        self.parser.add_argument('--style_dim', default=512, type=int,
                                 help='Input Style dimension')
        self.parser.add_argument('--n_mlp', default=8, type=int,
                                 help='Number of MLP layers')

        # Losses
        self.parser.add_argument('--d_reg_every', default=16, type=int,
                                 help="Regularize Discriminator after specified interval")
        self.parser.add_argument('--g_reg_every', default=4, type=int,
                                 help="Regularize Generator after specified interval")
        self.parser.add_argument('--lambda_disc', default=2, type=int,
                                 help="Multiplier factor for Discriminator loss")
        self.parser.add_argument('--lambda_id', default=1.2, type=int,
                                 help="Multiplier factor for id loss")
        self.parser.add_argument('--lambda_l2', default=10, type=int,
                                 help="Multiplier factor for smooth L2 loss")
        self.parser.add_argument('--lambda_lpips', default=2, type=int,
                                 help="Multiplier factor for lpips loss")
        self.parser.add_argument('--lambda_exp', default=0, type=int,
                                 help="Multiplier factor for Expression loss")
        self.parser.add_argument('--lambda_exp_warp', default=0, type=int,
                                 help="Multiplier factor for Expression Warp Loss")
        self.parser.add_argument('--lambda_path_loss', default=0, type=int,
                                 help="Multiplier factor for StyleGan path Loss")

        # Checkpoint paths
        self.parser.add_argument('--stylegan_weights', default="weights/ffhq-512-avg-tpurun1.pt", type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--load_stylegan_weights', default=True, type=bool,
                                 help='Load stylegan weights or not.')
        self.parser.add_argument('--retinaface_weights', default="weights/model_ir_se50.pth", type=str,
                                 help='Path to RetinaFace weights')
        self.parser.add_argument('--deca_weights', default="weights/deca_model.tar", type=str,
                                 help='Path to DECA weights')

        # RetinaFace
        self.parser.add_argument('--rf_size', default=112, type=int,
                                 help="RetinaFace model input size")
        self.parser.add_argument('--rf_num_layers', default=50, type=int,
                                 help="RetinaFace model number of layers")
        self.parser.add_argument('--rf_drop_ratio', default=0.6, type=int,
                                 help="RetinaFace model drop ratio")
        self.parser.add_argument('--rf_model_name', default="ir_se", type=str,
                                 help="RetinaFace backbone name")

        # VAE
        self.parser.add_argument('--vae_input_dim', default=50, type=int,
                                help="VAE input dimension")
        self.parser.add_argument('--vae_layer_dim', default=1024, type=int,
                                 help="VAE hidden layers dimensions")
        self.parser.add_argument('--vae_latent_dim', default=18, type=int,
                                 help="VAE latent dimension")
        self.parser.add_argument('--vae_num_layers', default=3, type=int,
                                 help="Number of VAE hidden layers")
        self.parser.add_argument('--vae_weights', default="weights/vae.ckpt", type=str,
                                 help="Path to VAE weights")

        # Training parameter
        self.parser.add_argument('--max_epoch', default=30, type=int,
                                 help='Maximum number of training epochs')
        self.parser.add_argument('--image_interval', default=500, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--training_stage', default="imitate", type=str,
                                 help='Training stage name')
        self.parser.add_argument('--checkpoint_path', default="checkpoints", type=str,
                                 help='Location of latest checkpoint')
        self.parser.add_argument('--latest_checkpoint', default="epoch=4-step=15439.ckpt", type=str,
                                 help='Location of latest checkpoint')


    def parse(self):
        opts = self.parser.parse_args()
        return opts
