import argparse
import os
from vedo import Volume, Axes, Box, settings, show
from vedo.applications import RayCastPlotter

def visualize_nifti(path_to_file, bg=(1,1,1), mesh_color=(1,0,0)):
    # Load NIfTI file
    vol = Volume(path_to_file)

    vol.mode(1).cmap("jet")  # change visual properties

    # Create a Plotter instance and show
    plt = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)
    plt.show(viewup="z")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize a NIfTI file.')
    parser.add_argument('--niftii_path', type=str, help='Path to the NIfTI file.')
    # optional arguments to load mask by patient and date
    parser.add_argument('--patient', type=str, help='Patient ID.')
    parser.add_argument('--date', type=str, help='Date of the scan.')
    parser.add_argument('--sequence', type=str, help='Sequence of the scan.')

    args = parser.parse_args()
    # if patient and date are given, define the path as follows
    if not args.sequence:
        args.sequence = 't1ce'

    if args.patient and args.date:
        args.niftii_path = os.path.join('//vf-DataSafe/DataSafe$/div2/radi/Brughoek_predict_1255/01_followup_cleanedup/followup_part1/', 'id_' + args.patient, args.date, args.patient + '_' + args.date + '_' + args.sequence + '_seg.nii.gz')

    print('Displaying segmentation file:' + args.niftii_path)
    visualize_nifti(args.niftii_path)
