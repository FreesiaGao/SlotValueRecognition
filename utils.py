# -*- coding:utf-8 -*-

import config as cfg


def get_chunks(seq, labels):
    """
    given a sequence of tags, group entities and their position
    :param seq: [4, 4, 0, 0, ...] sequence of labels
    :param tags: dict['0'] = 4
    :return: list of (chunk_type, chunk_start, chunk_end)
    example:
        seq = [4, 5, 0, 3]
        tags = {'B_PER': 4, 'I_PER': 5, 'B_LOC': 3}
        result = [('PER', 0, 2), ('LOC', 3, 4)]
    """
    default = labels[cfg.Config.default_label]
    idx_to_label = {idx: label for label, idx in labels.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok != default:
            if chunk_type is None and chunk_start is None:      # start of a chunk
                chunk_type = idx_to_label[tok]
                chunk_start = i
            elif idx_to_label[tok] == chunk_type:       # inside of a chunk
                pass
            else:       # end of a chunk and start of a new chunk
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = idx_to_label[tok], i

        else:
            if chunk_type is not None and chunk_start is not None:      # end of a chunk
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

    # a chunk at the end of the seq
    if chunk_type is not None and chunk_start is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
