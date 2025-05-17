#inspired from official github: https://github.com/mlech26l/ncps/blob/master/misc/ncp_cnn.png and their wormnet convolutional head
import torch
import torch.nn as nn
import torch.nn.functional as F

#how it works example: (text based visualisation was AI generated based off exact conv construction)
# Input [batch, seq_len, 160, 20, 3]
#   │
#   ↓
# Conv Layers (6 sequential layers with different configs)
#   │
#   ↓
# Final Feature Maps [batch*seq, 8, 7, 1]
#   │
#   ↓
# Split into 8 separate maps
#   │
#   ├─→ Map 1 [batch*seq, 1, 7, 1] → Flatten → Dense → 8 features ─┐
#   ├─→ Map 2 [batch*seq, 1, 7, 1] → Flatten → Dense → 8 features ─┤
#   ├─→ ...                                                          ├─→ Concatenate → 64 features
#   └─→ Map 8 [batch*seq, 1, 7, 1] → Flatten → Dense → 8 features ─┘
#   │
#   ↓
# Reshape to [seq_len, batch, 64] for LTC network


#---------------------------
# custom CNN head - create a convolutional head for feature extraction before passing into LTC
#---------------------------
class ConvHead(nn.Module):
    def __init__(self, 
                 #the output shape of the conv layer is determined by num_filters * features_per_filter, i.e default case feature output of shape (64,) or (batch, seq_len, 64,)
                 num_filters=8, #how many filters in final conv layer to feature extract from
                 features_per_filter=8, #how many features each num_filter layer shortens the original feature
                 img_h=160, #default img h, w and channels for comma ai dataset
                 img_w=320, 
                 channels=3):
        
        super(ConvHead, self).__init__() #inheret pytorch

        #store config variables
        self.num_filters = num_filters
        self.features_per_filter = features_per_filter 
        self.img_height = img_h 
        self.img_width = img_w 
        self.colour_channels = channels
        self.output_dim = num_filters * features_per_filter #expected output feature dimension (num of sensory, except num of sensory also includes features from speed embedding) 

        #create convolutional layers in tuple form (filter, kernel_size, stride)
        # self.conv_config = [(24,5,2), (36,5,2), (48,3,2), (64,3,1), (num_filters,3,1)]
        # self.conv_config = [(24,5,3), (36,5,2), (48,3,2), (64,3,1), (num_filters,3,1)] #increased stride
        self.conv_config = [(16,7,3), (24,5,2), (36,5,2), (48,3,2), (64,3,1), (num_filters,3,1)] #new downsample layer with increased stride
        self.conv_layers = nn.ModuleList() #empty parameter module, instead of nn.Sequential so we can manually select different layer for feature extraction to pass
        self.create_conv_layers() 
        self.create_feature_output_layer()
        self.init_weights()
    
    #weights in conv should already be initialised by default
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01) #add positive bias during init

    #creates dense layers to feature extract
    def create_feature_output_layer(self):
        self.feature_out_h, self.feature_out_w = self.calculate_conv_output_dim() #img starts in shape h x w and gets resized into a final shape h_f x w_f, the function is returning h_f, w_f to construct dense layer
        flattened_shape = self.feature_out_h * self.feature_out_w
        self.dense_layers = nn.ModuleList() #stores all num_filters amount of dense layers to turn final feature vector into shape (num_filters x features_per_filter) which gets concatennated into (features, )
        
        #create individual dense layers for num_filters
        for _ in range(self.num_filters):
            self.dense_layers.append(
                nn.Linear(flattened_shape, self.features_per_filter, bias=True) #linear dense layer with shape flattened (h_f x w_f) as input and output shape of features_per_filter (each linear layer outputs 4 dim feature vec)
            )
         
    def create_conv_layers(self):
        #create layers
        _channels = self.colour_channels
        for filter_amount, kernel_size, stride in self.conv_config:
            self.conv_layers.append(
                nn.Conv2d(in_channels=_channels, out_channels=filter_amount, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True)
             )
            _channels = filter_amount #dynamically update channels in each layer to the filter amount, as output from every conv layer is in shape ((img_h, img_w) // stride, filter_amount)
    
    #calculatyes expected output dim based on img_h, img_w and channels to initialise nn.Linear input shape (this dense layer takes the h x w as a single feature vector)
    def calculate_conv_output_dim(self):
        h, w = self.img_height, self.img_width

        for _, kernel_size, stride in self.conv_config: 
            padding = kernel_size // 2
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1

        return h, w
    
    def forward(self, x): #expects shape as B x seq_len x H x W x C
        batch_size, seq_len = x.size(0), x.size(1)
        
        #reshape to process all frames at once
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #(batch*seq), h, w, c
        
        #convert to NCHW format for pytorch convolutions 
        x = x.permute(0, 3, 1, 2)  #(batch*seq), c, h, w
        
        #normalise input using mean and std of each sample
        x = (x - x.mean(dim=(1,2,3), keepdim=True)) / (x.std(dim=(1,2,3), keepdim=True) + 1e-5)
        
        #apply convolution layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        #split filters along channel dimension
        filter_outputs = torch.split(x, 1, dim=1)  #list of tensors
        
        #extract features from each filter
        feature_layers = []
        for i, filter_output in enumerate(filter_outputs):
            #flatten the spatial dimensions
            flattened = filter_output.view(filter_output.size(0), -1)
            #apply linear layer
            feature = F.relu(self.dense_layers[i](flattened))
            feature_layers.append(feature)
        
        #concatenate all features to a single vector
        feature_layer = torch.cat(feature_layers, dim=1)  #(batch*seq), feature_dim
        
        #reshape back to sequence format
        feature_layer = feature_layer.view(batch_size, seq_len, -1)  #batch, seq, feature_dim

        #reshape feature output to sequence format - time-major format (seq_len, batch, feature_dim)
        feature_layer = feature_layer.permute(1, 0, 2)  #seq, batch, feature_dim
        
        return feature_layer
    
    def verify_construction(self):
        print("ConvHead Architecture:")
        print(f"- Expected input: [batch, {self.img_height}, {self.img_width}, {self.colour_channels}] (NHWC format)")
        print(f"  (With optional sequence dimension: [batch, seq_len, {self.img_height}, {self.img_width}, {self.colour_channels}])")
        print(f"- Note: Internally converts to NCHW format for convolutions")

        print(f"- Number of convolutional layers: {len(self.conv_layers)}")
        
        print("- Convolutional layers:")
        for i, (filters, kernel, stride) in enumerate(self.conv_config):
            print(f"  Layer {i+1}: {filters} filters, {kernel}x{kernel} kernel, stride {stride}")
        
        # print(f"- Final feature map dimensions: {self.feature_out_h}x{self.feature_out_w}")
        print(f"- Features per filter: {self.features_per_filter}")
        print(f"- Total output features: {self.output_dim}")
    
    def get_output_dim(self):
        return self.output_dim


#testing and veryfing construction
def test_with_new_dims():
    #create model with new dimensions
    conv_head = ConvHead(img_h=160, img_w=320)
    conv_head.verify_construction()
    
    #check parameter count
    total_params = sum(p.numel() for p in conv_head.parameters())
    print(f"Total parameters: {total_params:,}")
    
    #create sample input
    batch_size, seq_len = 2, 4
    x = torch.rand(batch_size, seq_len, 160, 320, 3)
    
    #forward pass
    features = conv_head(x)
    print(f"Output shape: {features.shape}")
    
    return conv_head, total_params

# test_with_new_dims() #uncomment to run and test
