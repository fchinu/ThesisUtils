import ROOT
import yaml
import argparse

def GetROOTColor(color='kBlack'):
    '''
    Method to retrieve a ROOT color

    Parameters
    ----------

    - color: color according to ROOT TColor convention

    Returns
    ----------

    - ROOT color corresponding to input color

    '''
    cMapROOT = {'kBlack': ROOT.kBlack, 'kWhite': ROOT.kWhite, 'kGrey': ROOT.kGray,
                'kRed': ROOT.kRed, 'kBlue': ROOT.kBlue, 'kGreen': ROOT.kGreen,
                'kTeal': ROOT.kTeal, 'kAzure': ROOT.kAzure, 'kCyan': ROOT.kCyan,
                'kOrange': ROOT.kOrange, 'kYellow': ROOT.kYellow, 'kSpring': ROOT.kSpring,
                'kMagenta': ROOT.kMagenta, 'kViolet': ROOT.kViolet, 'kPink': ROOT.kPink}

    ROOTcolor = None
    for colorKey in cMapROOT:
        if colorKey in color:
            ROOTcolor = cMapROOT.get(colorKey)
            break
    if ROOTcolor:
        for shade in range(0, 11):
            if f' + {shade}' in color or f'+{shade}' in color:
                ROOTcolor += shade
                break
            elif f' - {shade}' in color or f'-{shade}' in color:
                ROOTcolor -= shade
                break

    return ROOTcolor

def GetROOTMarker(marker='kFullCircle'):
    '''
    Method to retrieve the ROOT marker map

    Parameters
    ----------

    - color: color according to ROOT TColor convention

    Returns
    ----------

    - ROOT color corresponding to input color

    '''
    mMapROOT = {'kFullCircle': ROOT.kFullCircle, 'kFullSquare': ROOT.kFullSquare, 'kFullDiamond': ROOT.kFullDiamond,
                'kFullCross': ROOT.kFullCross, 'kFullTriangleUp': ROOT.kFullTriangleUp, 'kFullTriangleDown': ROOT.kFullTriangleDown,
                'kOpenCircle': ROOT.kOpenCircle, 'kOpenSquare': ROOT.kOpenSquare, 'kOpenDiamond': ROOT.kOpenDiamond,
                'kOpenCross': ROOT.kOpenCross, 'kOpenTriangleUp': ROOT.kOpenTriangleUp, 'kOpenTriangleDown': ROOT.kOpenTriangleDown}

    if marker in mMapROOT:
        ROOTmarker = mMapROOT.get(marker)
    else:
        ROOTmarker = None

    return ROOTmarker

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_canvas(width, height, title, margin):
    canvas = ROOT.TCanvas("canvas", title, width, height)
    canvas.SetLeftMargin(margin['left'])
    canvas.SetRightMargin(margin['right'])
    canvas.SetTopMargin(margin['top'])
    canvas.SetBottomMargin(margin['bottom'])
    return canvas

def add_histogram(file_name, histogram_name, color, marker_style, marker_size, scale_factor):
    file = ROOT.TFile(file_name)
    histogram = file.Get(histogram_name)
    histogram.SetDirectory(0)
    file.Close()

    if color:
        histogram.SetLineColor(GetROOTColor(color))
        histogram.SetMarkerColor(GetROOTColor(color))
    if marker_style:
        histogram.SetMarkerStyle(GetROOTMarker(marker_style))
    if marker_size:
        histogram.SetMarkerSize(marker_size)
    if scale_factor:
        histogram.Scale(scale_factor)

    return histogram

def add_text(latex_text, x, y, size, font):
    latex = ROOT.TLatex()
    latex.SetTextSize(size)
    latex.SetTextFont(font)
    latex.DrawLatexNDC(x, y, latex_text)

def add_legend(x1, y1, x2, y2, size, entries):
    legend = ROOT.TLegend(x1, y1, x2, y2)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(size)
    for entry in entries:
        legend.AddEntry(entry['histogram'], entry['label'], entry['style'])
    return legend

def draw_horizontal_line(y_value, line_color, line_style):
    line = ROOT.TLine()
    line.SetLineColor(GetROOTColor(line_color))
    line.SetLineStyle(line_style)
    line.DrawLine(ROOT.gPad.GetUxmin(), y_value, ROOT.gPad.GetUxmax(), y_value)

def main():
    parser = argparse.ArgumentParser(description='Generate ROOT figures from a YAML configuration file.')
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML configuration file', required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    canvas = create_canvas(config['canvas']['width'], config['canvas']['height'], config['canvas']['title'], config['canvas']['margin'])
    canvas.cd()

    frame = canvas.DrawFrame(
        config['canvas']['axes']['x']['range'][0],
        config['canvas']['axes']['y']['range'][0],
        config['canvas']['axes']['x']['range'][1],
        config['canvas']['axes']['y']['range'][1],
        config['canvas']['title'] + "; " + config['canvas']['axes']['x']['title']['name'] + "; " + config['canvas']['axes']['y']['title']['name'] + ";"
    )
    frame.GetXaxis().SetTitleSize(config['canvas']['axes']['x']['title']['size'])
    frame.GetYaxis().SetTitleSize(config['canvas']['axes']['y']['title']['size'])
    frame.GetXaxis().SetTitleOffset(config['canvas']['axes']['x']['title']['offset'])
    frame.GetYaxis().SetTitleOffset(config['canvas']['axes']['y']['title']['offset'])
    frame.GetXaxis().SetLabelSize(config['canvas']['axes']['x']['labels']['size'])
    frame.GetYaxis().SetLabelSize(config['canvas']['axes']['y']['labels']['size'])
    frame.GetXaxis().SetLabelOffset(config['canvas']['axes']['x']['labels']['offset'])
    frame.GetYaxis().SetLabelOffset(config['canvas']['axes']['y']['labels']['offset'])

    for input_data in config['input_data']:
        histogram = add_histogram(input_data['file'], input_data['histogram'], input_data['color'], input_data['marker_style'], input_data['marker_size'], input_data['scale_factor'])
        histogram.Draw("same")

    for text in config.get('text', []):
        add_text(text['content'], text['x'], text['y'], text['size'], text['font'])

    legend_config = config.get('legend')
    if legend_config['draw']:
        legend = add_legend(legend_config['x1'], legend_config['y1'], legend_config['x2'], legend_config['y2'], legend_config['size'], legend_config['entries'])
        legend.Draw("same")

    horizontal_line_config = config.get('horizontal_line')
    if horizontal_line_config['draw']:
        draw_horizontal_line(horizontal_line_config['height'], horizontal_line_config['line_color'], horizontal_line_config['line_style'])

    canvas.SaveAs(config['output_file'])

if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)  # Run in batch mode to suppress GUI
    main()
