module NumMethods

# load files
include("./GenModel/PFI.jl")
include("./ToyModel/VFI.jl")
include("./ToyModel/EGM.jl")
include("./ToyModel/PFI.jl")

# export
export ToyPFI, ToyEGM, ToyVFI, GenPFI

end
