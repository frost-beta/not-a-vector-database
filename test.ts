import {describe, it} from 'node:test';
import * as assert from 'node:assert';
import {core as mx} from '@frost-beta/mlx';

import {Storage} from './index.js';

function generateEmbedding() {
  const embeddingSize = 16;
  return mx.random.uniform(-1, 1, [ embeddingSize ]).toTypedArray() as Float32Array;
}

describe('Storage', () => {
  it('push', () => {
    const storage = new Storage();
    storage.push({embedding: generateEmbedding(), data: 1});
    assert.deepEqual(storage.embeddings.shape, [ 1, 16 ]);
    storage.push({embedding: generateEmbedding(), data: 2},
                 {embedding: generateEmbedding(), data: 3});
    assert.deepEqual(storage.embeddings.shape, [ 3, 16 ]);
    assert.deepEqual(storage.data, [ 1, 2, 3 ]);
  });

  it('search', () => {
    const size = 10;
    const storage = new Storage();
    for (let i = 0; i < size; ++i) {
      storage.push({embedding: generateEmbedding(), data: `data${i}`});
    }
    const results = storage.search(generateEmbedding());
    const scores = results.map(r => r.score);
    assert.ok(scores.every(s => s >= 0));
    assert.deepEqual(scores, scores.toSorted().toReversed());
  });

  it('serialization', () => {
    const size = 10;
    const storage = new Storage();
    const embeddings: Float32Array[] = [];
    const data: string[] = [];
    for (let i = 0; i < size; ++i) {
      embeddings.push(generateEmbedding());
      data.push(`data${i}`);
    }
    for (let i = 0; i < size; ++i) {
      storage.push({embedding: embeddings[i], data: data[i]});
    }
    storage.loadFromBuffer(storage.dumpToBuffer());
    for (let i = 0; i < size; ++i) {
      assert.deepEqual(storage.embeddings.index(i).tolist(), Array.from(embeddings[i]));
      assert.equal(storage.data[i], data[i]);
    }
  });
});
