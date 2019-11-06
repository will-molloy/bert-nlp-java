package com.wilmol.bert;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Verify.verify;

import com.google.common.primitives.Longs;
import java.util.List;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.tensorflow.example.Feature;
import org.tensorflow.example.Features;
import org.tensorflow.example.Int64List;

/**
 * Feature extractor. Extracts features from the text's tokens according to the models vocab.
 *
 * <p>Based on: <a
 * href=https://github.com/google-research/bert/blob/master/extract_features.py>https://github.com/google-research/bert/blob/master/extract_features.py</a>
 *
 * @author wilmol
 * @author LaurenceTews
 */
public class FeatureExtractor {

  private final Logger log = LogManager.getLogger();

  private final Model model;

  private final Tokenizer tokenizer;

  public FeatureExtractor(Model model, Tokenizer tokenizer) {
    this.model = checkNotNull(model);
    this.tokenizer = checkNotNull(tokenizer);
  }

  /**
   * Extracts features from the text.
   *
   * <p>Specifically: {@code segment_ids}, {@code label_ids}, {@code input_ids}, and {@code
   * input_mask}.
   *
   * <p>Note: if the input text (number of words) exceeds the {@code max_seq_length} of the model
   * the result will be truncated on the right.
   *
   * @param text text to extract features from
   * @return extracted features
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
        .putFeature("segment_ids", createFeatureFrom(new long[model.maxSeqLength()]))
        .putFeature("label_ids", createFeatureFrom(1))
        .putFeature("input_ids", createFeatureFrom(inputIds))
        .putFeature("input_mask", createFeatureFrom(inputMasks))
        .build();
  }

  private static Feature createFeatureFrom(long... values) {
    Int64List list = Int64List.newBuilder().addAllValue(Longs.asList(values)).build();
    return Feature.newBuilder().setInt64List(list).build();
  }
}
