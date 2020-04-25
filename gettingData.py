import glob
import numpy as np
import wfdb


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

        ann = wfdb.rdann(e, 'atr')
        good = ['N']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
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
    print(np_op.shape)
