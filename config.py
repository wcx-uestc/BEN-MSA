class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        """
        basic config
        """
        self.dataset = 'MOSI'
        self.data_path = '/home/lab/fuziwang/OGM/dataset/' + self.dataset
        self.label_path = ''
        self.mode = 'train'
        self.n_train = 0
        self.n_valid = 0
        self.n_test = 0
        self.batch_size = 32
        self.num_epochs = 40
        self.lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_decay_step = 20
        self.lr_decay_ratio = 0.1
        self.embed_dim = 32
        self.num_heads = 8
        self.num_self_attn = 4
        self.attn_dropout = 0.1
        self.relu_dropout = 0.1
        self.res_dropout = 0.1,
        self.embed_dropout = 0.1
        self.seq = 4 # size of bottleneck
        self.recursion = 4 # fuse num layers
        self.output_method = "sum"
        self.save_path = "final_model.pt"
        """
        OGM config
        """
        self.use_OGM = True
        self.use_GE = True
        self.alpha = 0.1
        """
        lora config
        """
        self.use_lora = False
        self.frozen = False
        self.r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.inference_mode = True
        """
        contrastive config
        """
        self.use_infonce = True
        self.embed_dropout = 0.1


def get_config(dataset, mode, batch_size):
    config = Config()
    config.dataset = dataset
    config.mode = mode
    config.batch_size = batch_size
    return config
