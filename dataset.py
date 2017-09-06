from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.spatial import ConvexHull


class DataGenerator(object):

    def __init__(self):
        """Construct a DataGenerator."""
        pass

    def next_batch(self, batch_size, N, train_mode=True, convex_hull=False):
        """Return the next batch of the data"""
        # If training on the convex hull problem: sequence of random points from [0, 1] x [0, 1]
        # If training on the sorting problem: sequence of random real numbers in [0, 1]
        reader_input_batch = []

        # Sorted sequence that we feed to encoder
        # In inference we feed an unordered sequence again
        decoder_input_batch = []

        # Ordered sequence where one hot vector encodes position in the input array
        writer_outputs_batch = []

        if convex_hull:
            for _ in range(N):
                reader_input_batch.append(np.zeros([batch_size, 2]))
            for _ in range(N+1):
                decoder_input_batch.append(np.zeros([batch_size, 2]))
                writer_outputs_batch.append(np.zeros([batch_size, N + 1]))

            for b in range(batch_size):
                sequence = np.random.rand(N, 2)
                leftmost_point = np.argmin(sequence[:,0])
                hull = ConvexHull(sequence)
                v = hull.vertices
                v = np.roll(v, -list(v).index(leftmost_point)) #start from leftmost point
                for i in range(N):
                    reader_input_batch[i][b] = sequence[i]

                for i in range(len(v)):
                    if train_mode:
                        decoder_input_batch[i + 1][b] = sequence[v[i]]
                    else:
                        decoder_input_batch[i + 1][b] = sequence[i]
                    writer_outputs_batch[i][b, v[i]+1] = 1.0

                #Write the stop symbol    
                for i in range(len(v), N):
                    writer_outputs_batch[i][b, 0] = 1.0
                    if not train_mode:
                        decoder_input_batch[i + 1][b] = sequence[i]
                writer_outputs_batch[N][b, 0] = 1.0
        else:
            
            for _ in range(N):
                reader_input_batch.append(np.zeros([batch_size, 1]))
            for _ in range(N + 1):
                decoder_input_batch.append(np.zeros([batch_size, 1]))
                writer_outputs_batch.append(np.zeros([batch_size, N + 1]))

            for b in range(batch_size):
                shuffle = np.random.permutation(N)
                sequence = np.sort(np.random.random(N))
                shuffled_sequence = sequence[shuffle]

                for i in range(N):
                    reader_input_batch[i][b] = shuffled_sequence[i]
                    if train_mode:
                        decoder_input_batch[i + 1][b] = sequence[i]
                    else:
                        decoder_input_batch[i + 1][b] = shuffled_sequence[i]
                    writer_outputs_batch[shuffle[i]][b, i + 1] = 1.0

                # Points to the stop symbol
                writer_outputs_batch[N][b, 0] = 1.0


        return reader_input_batch, decoder_input_batch, writer_outputs_batch
if __name__ == "__main__":
    dataset = DataGenerator()
    r, d, w = dataset.next_batch(1, 5, train_mode=False, convex_hull=True)
    print("Reader: ", r)
    print("Decoder: ", d)
    print("Writer: ", w)
