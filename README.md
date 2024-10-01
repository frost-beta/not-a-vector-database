# You need a few lines of JS, not a vector database

Suppose you are implementing RAG for your AI app, and you have used web APIs or
an inference engine to generate a large number of embeddings, and now you need
to find out the best results matching a query embedding, what do you do?

Use a vector database? No, you only need a few lines of JavaScript.

## A quick introduction to node-mlx

[MLX](https://github.com/ml-explore/mlx) is a full-featured machine learning
framework, with easy-to-understand source code and
[small binary sizes](https://github.com/frost-beta/node-mlx/releases).
And [node-mlx](https://github.com/frost-beta/node-mlx) is the JavaScript binding
of it.

MLX only has GPU support for macOS, and but its CPU support, implemented with
vectorized instructions, is still fast on Linux.

```typescript
import {core as mx, nn} from '@frost-beta/mlx';
```

## Embeddings search, in a few lines of JS

Suppose you want to find out the results with highest similaries to `query`,
from the `embeddings`.

```typescript
const embeddings = [
    [  0.037035,  0.0760545, ... ],
    [  0.034029, -0.0227216, ... ],
    ...
    [ -0.028612,  0.0052857, ... ],
];
const query = [ -0.019773, 0.006021, ... ];
```

With node-mlx, you can use the builtin `nn.losses.cosineSimilarityLoss` API to
do the search.

```typescript
const embeddingsTensor = mx.array(embeddings);
const queryTensor = mx.array([ query ]);
const scoresTensor = nn.losses.cosineSimilarityLoss(queryTensor, embeddingsTensor);
const scores: Float32Array = scoresTensor.toTypedArray();
```

The `scores` array stores the
[cosine similarities](https://en.wikipedia.org/wiki/Cosine_similarity)
between the `query` and `embeddings`.

(If you are wondering how we can compute cosine similarities between a 1x1
tensor and a 1xN tensor, it is called
[broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).)

```typescript
console.log(scores);
// [ 0.233244, 0.012492, ..., 0.43298 ]
```

## Sorting

Once you get the `scores` array, you can use the usual JavScript code to filter
and sort the results. But you can also use MLX if the number of results is large
enough to make JavScript engine struggle.

```typescript
// Get the top 10 scores.
const topTen = mx.topk(scoresTensor, 10);
console.log(topTen.toTypedArray());
// [ 0.894323, 0.594729, ... ]

// Sort the scores.
let sortedScores = mx.sort(scoresTensor);
console.log(sortedScores.toTypedArray());
// [ 0.01287, 0.1502876, ... ]
sortedScores = sortedScores.index(mx.Slice(null, null, -1));
console.log(sortedScores.toTypedArray());
// [ 0.894323, 0.594729, ... ]

// Get the indices of the scores ordered by their values in the array.
const indices = mx.argsort(scoresTensor)
    .index(mx.Slice(null, null, -1))
    .toTypedArray();
console.log(indices);
// [ 8, 9, ... ]
console.log(indices.map(i => scores[i]));
// [ 0.894323, 0.594729, ... ]
```

The `array.index(mx.Slice(null, null, -1))` code looks alien, it is actually the
JavaScript version of Python's `array[::-1]`, which reverse the array. You can
of course convert the result to JavaScript Array frist and then call
`reverse()`, but it would be slower if the array is very large.

## A Node.js module

If after reading above introductions you still find MXL cumbersome to use (which
is normal if you had zero experience with NumPy or PyTorch), I have wrapped the
code into a very simple Node.js module, which you can use to replace vector
databases in many cases.

Install:

```console
npm install not-a-vector-database
```

APIs:

```typescript
export type EmbeddingInput = number[] | TypedArray;

export interface SearchOptions {
    /**
     * Results with scores larger than this value will be returned. Should be
     * between -1 and 1. Default is 0.
     */
    minimumScore?: number;
    /**
     * Restrict the number of results to return, default is 16.
     */
    maximumResults?: number;
}

export interface SearchResult {
    score: number;
    data: unknown;
}

/**
 * In-memory storage of embeddings and associated data.
 */
export class Storage {
    embeddings?: mx.array;
    data: unknown[];

    /**
     * Initialize from the buffer.
     */
    loadFromBuffer(buffer: Buffer): void;

    /**
     * Dump the data to a buffer.
     */
    dumpToBuffer(): Buffer;

    /**
     * Add data to the storage.
     */
    push(...items: {embedding: EmbeddingInput; data: unknown;}[]): void;

    /**
     * Return the data which are most relevant to the embedding.
     */
    search(embedding: EmbeddingInput, options?: SearchOptions): SearchResult[];
}
```

Example:

```typescript
import {Storage} from 'not-a-vector-database';

const storage = new Stroage();
storage.push({embedding, data: 'some data'});

const results = storage.search(embedding);

fs.writeFileSync('storage.bser', storage.dumpToBuffer());
storage.loadFromBuffer(fs.readFileSync('storage.bser'));
```

There is also a `benchmark.ts` script that you can use to test the performance.
On my 2018 Intel MacBook Pro which has no GPU support, searching from 1 million
embeddings with size of 128 takes about 900ms.
