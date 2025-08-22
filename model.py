import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.cluster import KMeans
import importlib.util
from utils import reset_params,check_tensor

def antecedent_init_center(X, y=None, n_rule=2, method="kmean", engine="sklearn", n_init=20):
    """

    This function run KMeans clustering to obtain the :code:`init_center` for :func:`AntecedentGMF() <AntecedentGMF>`.

    Examples
    --------
    >>> init_center = antecedent_init_center(X, n_rule=10, method="kmean", n_init=20)
    >>> antecedent = AntecedentGMF(X.shape[1], n_rule=10, init_center=init_center)


    :param numpy.array X: Feature matrix with the size of :math:`[N,D]`, where :math:`N` is the
        number of samples, :math:`D` is the number of features.
    :param numpy.array y: None, not used.
    :param int n_rule: Number of rules :math:`R`. This function will run a KMeans clustering to
        obtain :math:`R` cluster centers as the initial antecedent center for TSK modeling.
    :param str method: Current version only support "kmean".
    :param str engine: "sklearn" or "faiss". If "sklearn", then the :code:`sklearn.cluster.KMeans()`
        function will be used, otherwise the :code:`faiss.Kmeans()` will be used. Faiss provide a
        faster KMeans clustering algorithm, "faiss" is recommended for large datasets.
    :param int n_init: Number of initialization of the KMeans algorithm. Same as the parameter
        :code:`n_init` in :code:`sklearn.cluster.KMeans()` and the parameter :code:`nredo` in
        :code:`faiss.Kmeans()`.
    """
    def faiss_cluster_center(X, y=None, n_rule=2, n_init=20):
        import faiss
        km = faiss.Kmeans(d=X.shape[1], k=n_rule, nredo=n_init)
        km.train(np.ascontiguousarray(X.astype("float32")))
        centers = km.centroids.T
        return centers

    if method == "kmean":
        if engine == "faiss":
            package_name = "faiss"
            spec = importlib.util.find_spec(package_name)
            if spec is not None:
                center = faiss_cluster_center(X=X, y=y, n_rule=n_rule)
                return center
            else:
                print("Package " + package_name + " is not installed, running scikit-learn KMeans...")
        km = KMeans(n_clusters=n_rule, n_init=n_init)
        km.fit(X)
        return km.cluster_centers_.T


class Antecedent(nn.Module):
    def forward(self, **kwargs):
        raise NotImplementedError

    def init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError


class AntecedentGMF(Antecedent):
    """

    Parent: :code:`torch.nn.Module`

    The antecedent part with Gaussian membership function. Input: data, output the corresponding
    firing levels of each rule. The firing level :math:`f_r(\mathbf{x})` of the
    :math:`r`-th rule are computed by:

    .. math::
        &\mu_{r,d}(x_d) = \exp(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}),\\
        &f_{r}(\mathbf{x})=\prod_{d=1}^{D}\mu_{r,d}(x_d),\\
        &\overline{f}_r(\mathbf{x}) = \frac{f_{r}(\mathbf{x})}{\sum_{i=1}^R f_{i}(\mathbf{x})}.


    :param int in_dim: Number of features :math:`D` of the input.
    :param int n_rule: Number of rules :math:`R` of the TSK model.
    :param bool high_dim: Whether to use the HTSK defuzzification. If :code:`high_dim=True`,
        HTSK is used. Otherwise the original defuzzification is used. More details can be found at [1].
        TSK model tends to fail on high-dimensional problems, so set :code:`high_dim=True` is highly
         recommended for any-dimensional problems.
    :param numpy.array init_center: Initial center of the Gaussian membership function with
        the size of :math:`[D,R]`. A common way is to run a KMeans clustering and set
        :code:`init_center` as the obtained centers. You can simply run
        :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>`
        to obtain the center.
    :param float init_sigma: Initial :math:`\sigma` of the Gaussian membership function.
    :param float eps: A constant to avoid the division zero error.
    """
    def __init__(self, in_dim, n_rule, high_dim=False, init_center=None, init_sigma=1., eps=1e-8):
        super(AntecedentGMF, self).__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.high_dim = high_dim

        self.init_center = check_tensor(init_center, torch.float32) if init_center is not None else None
        self.init_sigma = init_sigma
        self.zr_op = torch.mean if high_dim else torch.sum
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        self.center = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))
        self.sigma = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))

        self.reset_parameters()

    def init(self, center, sigma):
        """

        Change the value of :code:`init_center` and :code:`init_sigma`.

        :param numpy.array center: Initial center of the Gaussian membership function with the
            size of :math:`[D,R]`. A common way is to run a KMeans clustering and set
            :code:`init_center` as the obtained centers. You can simply run
            :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>`
            to obtain the center.
        :param float sigma: Initial :math:`\sigma` of the Gaussian membership function.
        """
        center = check_tensor(center, torch.float32)
        self.init_center = center
        self.init_sigma = sigma

        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all parameters.

        :return:
        """
        init.constant_(self.sigma, self.init_sigma)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)
        else:
            init.normal_(self.center, 0, 1)

    def forward(self, X):
        """

        Forward method of Pytorch Module.

        :param torch.tensor X: Pytorch tensor with the size of :math:`[N, D]`, where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R]`.
        """
        frs = self.zr_op(
            -(X.unsqueeze(dim=2) - self.center) ** 2 * (0.5 / (self.sigma ** 2 + self.eps)), dim=1
        )
        frs = F.softmax(frs, dim=1)
        return frs
class TSK(nn.Module):
    """

    Parent: :code:`torch.nn.Module`

    This module define the consequent part of the TSK model and combines it with a pre-defined
     antecedent module. The input of this module is the raw feature matrix, and output
     the final prediction of a TSK model.

    :param int in_dim: Number of features :math:`D`.
    :param int out_dim: Number of output dimension :math:`C`.
    :param int n_rule: Number of rules :math:`R`, must equal to the :code:`n_rule` of
        the :code:`Antecedent()`.
    :param torch.Module antecedent: An antecedent module, whose output dimension should be
        equal to the number of rules :math:`R`.
    :param int order: 0 or 1. The order of TSK. If 0, zero-oder TSK, else, first-order TSK.
    :param float eps: A constant to avoid the division zero error.
    :param torch.nn.Module consbn: If none, the raw feature will be used as the consequent input;
        If a pytorch module, then the consequent input will be the output of the given module.
        If you wish to use the BN technique we mentioned in
        `Models & Technique <../models.html#batch-normalization>`_,  you can set
        :code:`precons=nn.BatchNorm1d(in_dim)`.
    """
    def __init__(self, in_dim, out_dim, n_rule, antecedent, order=1, eps=1e-8, precons=None):
        super(TSK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rule = n_rule
        self.antecedent = antecedent
        self.precons = precons

        self.order = order
        assert self.order == 0 or self.order == 1, "Order can only be 0 or 1."
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        if self.order == 0:
            self.cons = nn.Linear(self.n_rule, self.out_dim, bias=True)
        else:
            self.cons = nn.Linear((self.in_dim + 1) * self.n_rule, self.out_dim)

    def reset_parameters(self):
        """
        Re-initialize all parameters, including both consequent and antecedent parts.

        :return:
        """
        reset_params(self.antecedent)
        self.cons.reset_parameters()

        if self.precons is not None:
            self.precons.reset_parameters()

    def forward(self, X, get_frs=False):
        """

        :param torch.tensor X: Input matrix with the size of :math:`[N, D]`,
            where :math:`N` is the number of samples.
        :param bool get_frs: If true, the firing levels (the output of the antecedent)
            will also be returned.

        :return: If :code:`get_frs=True`, return the TSK output :math:`Y\in \mathbb{R}^{N,C}`
            and the antecedent output :math:`U\in \mathbb{R}^{N,R}`. If :code:`get_frs=False`,
            only return the TSK output :math:`Y`.
        """
        frs = self.antecedent(X)

        if self.precons is not None:
            X = self.precons(X)

        if self.order == 0:
            cons_input = frs
        else:
            X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rule, X.size(1)])  # [n_batch, n_rule, in_dim]
            X = X * frs.unsqueeze(dim=2)
            X = X.view([X.size(0), -1])
            cons_input = torch.cat([X, frs], dim=1)

        output = self.cons(cons_input)
        if get_frs:
            return output, frs
        return output
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # Split the embedding into self.n_heads different heads
        query = query.view(query.shape[0], -1, self.n_heads, self.head_dim)
        key = key.view(key.shape[0], -1, self.n_heads, self.head_dim)
        value = value.view(value.shape[0], -1, self.n_heads, self.head_dim)

        # Transpose to get dimensions batch_size, self.n_heads, seq_len, self.head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Calculate the attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.nn.functional.softmax(scores, dim=-1)

        out = torch.matmul(attention, value)

        # Reshape to get back to the original input shape
        out = out.transpose(1, 2).contiguous().view(query.shape[0], -1, self.d_model)

        out = self.fc_out(out)
        return out

# Define the Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self,  d_model, n_heads, num_encoder_layers, dim_feedforward, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_encoder_layers)]
        )
        #self.src_mask = self.generate_square_subsequent_mask(max_seq_length)
        self.src_mask = None
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz) == 1, diagonal=1)).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        #src = src.permute(1, 0, 2)
        for layer in self.encoder:
            src = layer(src, self.src_mask)
        return src#.permute(1, 0, 2)
    
class TSK_ATT(nn.Module):
    def __init__(self, args):
        super(TSK_ATT, self).__init__()
        self.args = args
        self.max_seq_length = args.max_seq_length
        self.n_classes = args.n_classes
        if self.args.mode==1:
            input_dim2=input_dim3=input_dim4=args.input_dim
        elif self.args.mode==2:
            input_dim2=args.input_dim+args.n_classes
            input_dim3=args.input_dim+args.n_classes
            input_dim4=args.input_dim+args.n_classes
            input_dim5=args.input_dim+args.n_classes
            input_dim6=args.input_dim+args.n_classes
            input_dim7=args.input_dim+args.n_classes
            input_dim8=args.input_dim+args.n_classes
        elif self.args.mode==3:
            input_dim2=args.input_dim+args.n_classes
            input_dim3=input_dim2+args.n_classes
            input_dim4=input_dim3+args.n_classes
            # input_dim5=input_dim4+args.n_classes
        self.tsk1_antecedent=nn.Sequential(AntecedentGMF(in_dim=args.input_dim, n_rule=args.n_rule, high_dim=True, init_center=None),
                                     nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
       
        self.tsk2_antecedent = nn.Sequential(AntecedentGMF(in_dim=input_dim2, n_rule=args.n_rule, high_dim=True, init_center=None),
                                    nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
        self.tsk3_antecedent=nn.Sequential(AntecedentGMF(in_dim=input_dim3, n_rule=args.n_rule, high_dim=True, init_center=None),
                                     nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
       
        self.tsk4_antecedent = nn.Sequential(AntecedentGMF(in_dim=input_dim4, n_rule=args.n_rule, high_dim=True, init_center=None),
                                    nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
        self.tsk5_antecedent = nn.Sequential(AntecedentGMF(in_dim=input_dim5, n_rule=args.n_rule, high_dim=True, init_center=None),
                                    nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
        self.tsk6_antecedent = nn.Sequential(AntecedentGMF(in_dim=input_dim6, n_rule=args.n_rule, high_dim=True, init_center=None),
                                    nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
        self.tsk7_antecedent = nn.Sequential(AntecedentGMF(in_dim=input_dim7, n_rule=args.n_rule, high_dim=True, init_center=None),
                                    nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
        self.tsk8_antecedent = nn.Sequential(AntecedentGMF(in_dim=input_dim8, n_rule=args.n_rule, high_dim=True, init_center=None),
                                    nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
        self.tsk1=TSK(in_dim=args.input_dim, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.tsk1_antecedent, order=args.order, precons=None)
        self.tsk2=TSK(in_dim=input_dim2, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.tsk2_antecedent, order=args.order, precons=None)
        self.tsk3=TSK(in_dim=input_dim3, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.tsk3_antecedent, order=args.order, precons=None)
        self.tsk4=TSK(in_dim=input_dim4, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.tsk4_antecedent, order=args.order, precons=None)
        self.tsk5=TSK(in_dim=input_dim5, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.tsk5_antecedent, order=args.order, precons=None)
        self.tsk6=TSK(in_dim=input_dim6, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.tsk5_antecedent, order=args.order, precons=None)
        self.tsk7=TSK(in_dim=input_dim7, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.tsk5_antecedent, order=args.order, precons=None)
        self.tsk8=TSK(in_dim=input_dim8, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.tsk5_antecedent, order=args.order, precons=None)
        self.transformer = MultiHeadAttention( args.n_classes, args.n_heads)
    def forward(self,inputx):
        b,_ = inputx.shape[0],inputx.shape[1]
        #print(inputx.shape)
        
        tsk1 = self.tsk1(inputx)
        #print(tsk1.shape)
        tsk2_input = torch.cat([inputx,tsk1],dim=-1)
        tsk2 = self.tsk2(tsk2_input)
        tsk3_input = torch.cat([inputx,tsk2],dim=-1)
        tsk3 = self.tsk3(tsk3_input)
        tsk4_input = torch.cat([inputx,tsk3],dim=-1)
        tsk4 =self.tsk4(tsk4_input)
        tsk5_input = torch.cat([inputx,tsk4],dim=-1)
        tsk5 =self.tsk5(tsk5_input)     
        tsk6_input = torch.cat([inputx,tsk5],dim=-1)
        tsk6 =self.tsk6(tsk6_input)    
        tsk7_input = torch.cat([inputx,tsk6],dim=-1)
        tsk7 =self.tsk7(tsk7_input) 
        tsk8_input = torch.cat([inputx,tsk7],dim=-1)
        tsk8 =self.tsk8(tsk8_input) 
        transformer_input = torch.zeros((b,self.max_seq_length,tsk1.shape[1]))
        transformer_input[:,1,:]=tsk1#30+
        transformer_input[:,2,:]=tsk2#37
        transformer_input[:,3,:]=tsk3
        transformer_input[:,4,:]=tsk4
        transformer_input[:,5,:]=tsk5
        transformer_input[:,6,:]=tsk6
        transformer_input[:,7,:]=tsk7
        transformer_input[:,8,:]=tsk8
        output = self.transformer(transformer_input,transformer_input,transformer_input)[:,0,:]#b,l,f
        return output,tsk1,tsk2,tsk3,tsk4,tsk5,tsk6,tsk7,tsk8

        # ##仅用于推理时候，输出Z表征
        # Z = self.transformer(transformer_input, transformer_input, transformer_input)[:, 0, :]  # shape: (batch_size, class_dim)
        # self.output_z = Z  # 保存为属性
        # return Z, tsk1, tsk2, tsk3, tsk4

        
'''
# Instantiate the model
vocab_size = 10000  # Adjust as needed
d_model = 512
n_heads = 8
num_encoder_layers = 6
dim_feedforward = 2048
max_seq_length = 100
dropout = 0.1

model = Transformer(vocab_size, d_model, n_heads, num_encoder_layers, dim_feedforward, max_seq_length, dropout)
'''