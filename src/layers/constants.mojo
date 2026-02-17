from layout import Layout, LayoutTensor
from utils import Variant

struct LayerFuncTypeConstants[dtype: DType]:
    comptime LayerInputType = Variant[LayoutTensor[Self.dtype, Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE), MutAnyOrigin]]
    comptime LayerGradOutputType = Variant[LayoutTensor[Self.dtype, Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE), MutAnyOrigin]]
