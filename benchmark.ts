import {core as mx} from '@frost-beta/mlx';
import {Storage} from './index.js';

const embeddingSize = 128;
const totalDataSize = 1000 * 1000;

console.log('Preparing 1M embeddings...');
const embeddings = mx.random.uniform(-1, 1, [ totalDataSize, embeddingSize ]);
const query = mx.random.uniform(-1, 1, [ embeddingSize ]);
mx.eval(embeddings, query);

const storage = new Storage();
storage.embeddings = embeddings;
storage.data = new Array(totalDataSize);

console.log('Searching...');
const input = query.toTypedArray();
console.time('Time');
const results = storage.search(input);
console.timeEnd('Time');
