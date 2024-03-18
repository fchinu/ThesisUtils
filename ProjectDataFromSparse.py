import sys
import argparse
import yaml
import numpy as np
import uproot
sys.path.append("DmesonAnalysis")
from scipy.interpolate import InterpolatedUnivariateSpline
from ROOT import TFile, TH1F, TDatabasePDG # pylint: disable=import-error,no-name-in-module
from utils.TaskFileLoader import LoadNormObjFromTask, LoadSparseFromTask
from utils.DfUtils import FilterBitDf, LoadDfFromRootOrParquet
from utils.AnalysisUtils import MergeHists, ApplySplineFuncToColumn

parser = argparse.ArgumentParser(description='Arguments to pass')
parser.add_argument('sparseName', metavar='text', default='sparseName.root',
                    help='file name with root sparse')
parser.add_argument('cutSetFileName', metavar='text', default='cutSetFileName.yml',
                    help='input file with cut set')
parser.add_argument('outFileName', metavar='text', default='outFileName.root',
                    help='output root file name')
args = parser.parse_args()


# selections to be applied
with open(args.cutSetFileName, 'r') as ymlCutSetFile:
    cutSetCfg = yaml.load(ymlCutSetFile, yaml.FullLoader)

# Open the file
file = TFile.Open(args.sparseName)
# Get the THnSparse
sparse = file.Get("hf-task-ds/hSparseMass")

# Extract variable names and cuts from the config
var_names = list(cutSetCfg['cutvars'].keys())
var_names.remove('InvMass')
cuts = [list(zip(cutSetCfg['cutvars'][var]['min'], cutSetCfg['cutvars'][var]['max'])) for var in var_names]
axes = [cutSetCfg['cutvars'][var]['axisnum'] for var in var_names]

hMs = []
hPts = []

for iPt in range(len(cutSetCfg['cutvars']['Pt']['min'])):
    ptLowLabel = cutSetCfg['cutvars']['Pt']['min'][iPt] * 10
    ptHighLabel = cutSetCfg['cutvars']['Pt']['max'][iPt] * 10
    for i, axis in enumerate(axes):
        min = cuts[i][iPt][0]
        max = cuts[i][iPt][1]
        sparse.GetAxis(axis).SetRangeUser(min, max)
    hM = sparse.Projection(0)
    hM.SetDirectory(0)
    hPt = sparse.Projection(1)
    hPt.SetDirectory(0)
    hM.SetName(f'hMass_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
    hPt.SetName(f'hPt_{ptLowLabel:.0f}_{ptHighLabel:.0f}')
    hMs.append(hM)
    hPts.append(hPt)

outFile = TFile(args.outFileName, 'recreate')
outFile.cd()
for hM in hMs:
    hM.Write()
for hPt in hPts:
    hPt.Write()
outFile.Close()
