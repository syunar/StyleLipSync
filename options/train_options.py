from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        super().initialize(parser)
        # train options
        parser.add_argument("--iter",  default=200000, type=int, help="total training iteration")
        parser.add_argument('--batch_size', default=8, type=int, help='input batch size')
        parser.add_argument('--num_workers', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--resume', default=None, type=str, help="model path to resume")
       
        parser.add_argument('--Glr', default=0.0001, type=float, help='Generator learning rate')
        parser.add_argument('--Gbeta1', default=0.9,  type=float, help='Generator beta1 ')
        parser.add_argument('--Gbeta2', default=0.99, type=float, help='Generator beta2')
        
        parser.add_argument('--data', default='train', type=str, help='option for dataset')
        parser.add_argument('--dim_a', type=int, default=512, help='audio dimension')
        parser.add_argument('--dim_l', type=int, default=512, help='lip dimension')
        parser.add_argument('--dim_w', type=int, default=512, help='w latent dimension')
                
        parser.add_argument('--n_ref', type=int, default=1)
        parser.add_argument('--debug', action='store_true', help='use debug mode')
        
        # losses
        parser.add_argument('--lambda_l2', type=float, default=1)
        parser.add_argument('--lambda_lip', type=float, default=0)
        parser.add_argument('--lambda_lpips', type=float, default=1)
        parser.add_argument('--lambda_syncnet', type=float, default=0.1)
        parser.add_argument('--lambda_reg_sync', type=float, default=0)
        parser.add_argument('--log_img_seq', action='store_true')
        # log
        parser.add_argument('--print_freq', default=1, type=int, help='frequency of showing training results on console')
        parser.add_argument('--image_freq', default=100, type=int, help='frequency of saving the output images')
        parser.add_argument('--eval_freq', default=5000, type=int, help='frequency of evaluate model')
        parser.add_argument('--save_freq', default=5000, type=int, help='frequency of saving the model')

        # for distributed trainig
        parser.add_argument('--dist-url', default='2155', type=str, help='url for setting up distributed training')
        parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int, help='distributed backend')
        parser.add_argument('--dist-backend', default='nccl', type=str, help='node rank for distributed training')
        return parser
