import os
import sys
import imageio
from matplotlib import pyplot as plt
from tqdm import tqdm


def create_gif(filenames, duration, output_folder):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = output_folder + 'result.gif'
    imageio.mimsave(output_file, images, duration=duration)


if __name__ == "__main__":
    duration = 0.05
    labels = ['core']  # os.listdir('/home/featurize/work/finalyearproject/Segmentation/log/visualise/')
    for label in labels:
        patients = os.listdir('/home/featurize/work/finalyearproject/Segmentation/log/visualise/' + label + '/')
        for patient in patients:
            folder = '/home/featurize/work/finalyearproject/Segmentation/log/visualise/' + label + '/' + patient + '/'
            ground_truths = [file for file in os.listdir(folder) if 'ground_truth' in file]
            source_outputs = [file for file in os.listdir(folder) if 'source' in file]
            target_outputs = [file for file in os.listdir(folder) if 'target' in file]
            assert len(ground_truths) == len(source_outputs) == len(target_outputs)
            # Merge 3 figures together.
            for i in tqdm(range(len(ground_truths))):
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(7, 2))
                ax.flat[0].imshow(imageio.imread(folder + ground_truths[i]))
                ax.flat[0].set_title('Ground Truth')
                ax.flat[1].imshow(imageio.imread(folder + source_outputs[i]))
                ax.flat[1].set_title('Source Outputs')
                ax.flat[2].imshow(imageio.imread(folder + target_outputs[i]))
                ax.flat[2].set_title('Target Outouts')
                plt.savefig(folder + '%003d.jpg' % i)
                plt.close()
            # Convert to gif.
            filenames = [folder + file for file in os.listdir(folder) if len(file) == 7]
            create_gif(filenames, duration, folder)

    # create_gif(filenames, duration)