import pytest
import numpy as np

from mimic.datatype import CommandDataChunk

@pytest.fixture(scope="module")
def cmd_datachunk():
    chunk = CommandDataChunk()
    for i in range(10):
        seq = np.zeros((20, 7))
        chunk.push_epoch(seq)
    return chunk

def test_featureseq_list_generation_pipeline(cmd_datachunk):
    fslist = cmd_datachunk.to_featureseq_list()
    assert len(fslist) == 10
    assert list(fslist[0].size()) == [20, 7]
