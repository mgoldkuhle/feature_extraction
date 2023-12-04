import argparse
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
    parser.add_argument('niftii_path', type=str, help='Path to the NIfTI file.')

    args = parser.parse_args()

    visualize_nifti(args.niftii_path)
