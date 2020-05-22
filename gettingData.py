import glob
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import cv2


def get_records():
    """ Get paths for data in data/mit/ directory """
    # Download if doesn't exist

    # There are 3 files for each record
    # *.atr is one of them

    paths = glob.glob('../mit-bih-arrhythmia-database-1.0.0/*.atr')

    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    paths.sort()
    print(paths)

    return paths


def beat_annotations(annotation):
    """ Get rid of non-beat markers """
    """'N' for normal beats. Similarly we can give the input 'L' for left bundle branch block beats. 'R' for right bundle branch block
        beats. 'A' for Atrial premature contraction. 'V' for ventricular premature contraction. '/' for paced beat. 'E' for Ventricular
        escape beat."""

    good = ['N']
    ids = np.in1d(annotation.symbol, good)

    # We want to know only the positions
    beats = annotation.sample[ids]

    return beats


def segmentation(records):

    Normal = []
    for e in records:
        print(e)
        signals, fields = wfdb.rdsamp(e, channels=[0])
        print(signals)
        print(fields)

        ann = wfdb.rdann(e, 'atr')
        print(ann)
        good = ['N']
        ids = np.in1d(ann.symbol, good)
        print(ids)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        print("imp_beats", imp_beats)
        for i in imp_beats:
            beats = list(beats)
            j = beats.index(i)
            if(j != 0 and j != (len(beats)-1)):
                x = beats[j-1]
                y = beats[j+1]
                diff1 = abs(x - beats[j])//2
                diff2 = abs(y - beats[j])//2
                Normal.append(signals[beats[j] - diff1: beats[j] + diff2, 0])
    return Normal


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    op = segmentation(get_records())
    np_op = np.array(op)
    directory = 'images'
    print(np_op.shape)
    print("here", op)
    for count, i in enumerate(op):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        filename = directory + '/' + str(count)+'.png'
        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (128, 128),
                             interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(filename, im_gray)
