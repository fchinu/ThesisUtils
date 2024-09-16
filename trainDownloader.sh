ISMC=1
INPUT_FOLDER="/alice/cern.ch/user/a/alihyperloop/outputs/0026/264446/40069"
TRAIN_NUMBER="264446"
OUTPUTNAME="LHC24_g5"
OUTPUTDIR="/home/fchinu/Run3/Ds_pp_13TeV/Datasets/Ds_pp_run3_ml"

if [ $ISMC -eq 1 ]; then
    OUTPUTDIR=$OUTPUTDIR"/MC/Train"$TRAIN_NUMBER
else
    OUTPUTDIR=$OUTPUTDIR"/Data/Train"$TRAIN_NUMBER
fi

# Create the output directory
mkdir -p $OUTPUTDIR

# Download the file
alien.py cp alien:$INPUT_FOLDER/AnalysisResults.root file:$OUTPUTDIR/AnalysisResults_$OUTPUTNAME.root
alien.py cp alien:$INPUT_FOLDER/AO2D.root file:$OUTPUTDIR/AO2D_$OUTPUTNAME.root
