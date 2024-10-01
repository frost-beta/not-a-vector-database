import {core as mx, nn} from '@frost-beta/mlx';
import * as bser from 'bser';

type TypedArray = Int8Array | Uint8Array | Int16Array | Uint16Array |
                  Int32Array | Uint32Array | Float32Array | Float64Array;
export type EmbeddingInput = number[] | TypedArray | mx.array;

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
  data: unknown[] = [];

  /**
   * Initialize from the buffer.
   */
  loadFromBuffer(buffer: Buffer) {
    const json = bser.loadFromBuffer(buffer) as any;
    if (!Array.isArray(json.embeddings) && !Array.isArray(json.data))
      throw new Error('The buffer does not includes valid data.');
    if (this.embeddings)
      mx.dispose(this.embeddings);
    this.embeddings = mx.array(json.embeddings);
    this.data = json.data;
  }

  /**
   * Dump the data to a buffer.
   */
  dumpToBuffer(): Buffer {
    if (!this.embeddings)
      throw new Error('There is no data in storage.');
    return bser.dumpToBuffer({
      embeddings: this.embeddings.tolist(),
      data: this.data,
    });
  }

  /**
   * Add data to the storage.
   */
  push(...items: {embedding: EmbeddingInput, data: unknown}[]) {
    // Make sure intermediate arrays are released.
    mx.tidy(() => {
      const newEmbeddings = mx.stack(items.map(i => mx.array(i.embedding)));
      if (!this.embeddings) {
        this.embeddings = newEmbeddings;
      } else {
        const old = this.embeddings;
        this.embeddings = mx.concatenate([ old, newEmbeddings ], 0);
        // Release the old embeddings object, which is not caught by tidy.
        mx.dispose(old);
      }
      // Do not release this.embeddings.
      return this.embeddings;
    });
    this.data.push(...items.map(i => i.data));
  }

  /**
   * Return the data which are most relevant to the embedding.
   */
  search(embedding: EmbeddingInput,
         {
           minimumScore = 0,
           maximumResults = 16,
         }: SearchOptions = {}): SearchResult[] {
    if (!this.embeddings)
      return [];
    return mx.tidy(() => {
      const query = mx.array(embedding, this.embeddings.dtype).index(mx.newaxis);
      const scores = nn.losses.cosineSimilarityLoss(query, this.embeddings);
      const indices = mx.argsort(scores).index(mx.Slice(null, null, -1))
                                        .index(mx.Slice(null, maximumResults))
                                        .toTypedArray();
      const results: SearchResult[] = [];
      for (const index of indices) {
        const score = scores.index(index).item() as number;
        if (score < minimumScore)
          break;
        results.push({score, data: this.data[index]});
      }
      return results;
    });
  }
}
