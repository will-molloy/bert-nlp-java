package com.wilmol.bert.features;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Verify.verify;

import com.wilmol.bert.Model;
import com.wilmol.bert.tokenizer.Tokenizer;
import java.util.List;
import org.tensorflow.example.Feature;
import org.tensorflow.example.Features;
import org.tensorflow.example.Int64List;

/**
 * Feature extractor. Based on: <a
 * href=https://github.com/google-research/bert/blob/master/extract_features.py>https://github.com/google-research/bert/blob/master/extract_features.py</a>
 *
 * @author wilmol
 */
public class FeatureExtractor {

  private final Model model;

  private final Tokenizer tokenizer;

  public FeatureExtractor(Model model, Tokenizer tokenizer) {
    this.model = checkNotNull(model);
    this.tokenizer = checkNotNull(tokenizer);
  }

  /**
   * Extracts features from the given text.
   *
   * <p>Specifically: {@code segment_ids}, {@code label_ids}, {@code input_ids}, and {@code
   * input_mask}.
   *
   * <p>Note if the given text produces more features than the models max sequence length, it'll be
   * truncated on the right.
   *
   * @param text text to extract features from.
   * @return extracted features.
   */
  public Features extractFeatures(String text) {
    List<String> tokens = tokenizer.tokenize(text);
    verify(
        tokens.size() <= model.maxSeqLength(),
        "tokens.size() > max_seq_length (%s)",
        model.maxSeqLength());

    long[] inputIds = new long[model.maxSeqLength()];
    long[] inputMasks = new long[model.maxSeqLength()];
    for (int i = 0; i < tokens.size(); i++) {
      inputIds[i] = model.vocab().get(tokens.get(i));
      inputMasks[i] = 1;
    }

    return Features.newBuilder()
        .putFeature("segment_ids", createFeature(new long[model.maxSeqLength()]))
        .putFeature("label_ids", createFeature(1))
        .putFeature("input_ids", createFeature(inputIds))
        .putFeature("input_mask", createFeature(inputMasks))
        .build();
  }

  private static Feature createFeature(long... values) {
    Int64List.Builder builder = Int64List.newBuilder();
    for (long value : values) {
      builder.addValue(value);
    }
    return Feature.newBuilder().setInt64List(builder).build();
  }
}
