module NumMethods

# load files
# include("GenModel/PFI.jl")
include("./ToyModel/VFI.jl")
include("./ToyModel/EGM.jl")
include("./ToyModel/PFI.jl")
# include("./Plotting/") # TODO: add here

# export
export ToyPFI, ToyEGM, ToyVFI
# export ToyVFI, GenPFI

end
