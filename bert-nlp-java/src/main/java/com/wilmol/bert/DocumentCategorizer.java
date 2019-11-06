package com.wilmol.bert;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Verify.verify;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Range;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.example.Example;
import org.tensorflow.example.Features;

/**
 * Interface to categorize (i.e. classify/label) text using a trained BERT model.
 *
 * @author LaurenceTews
 * @author wilmol
 */
public class DocumentCategorizer {

  private static final String INPUT_TENSOR_NAME = "foo/input_example_tensor";

  private static final String OUTPUT_TENSOR_NAME = "loss/Softmax";

  private final Logger log = LogManager.getLogger();

  private final Model model;

  private final FeatureExtractor featureExtractor;

  public DocumentCategorizer(Model model, FeatureExtractor featureExtractor) {
    this.model = checkNotNull(model);
    this.featureExtractor = checkNotNull(featureExtractor);
  }

  /**
   * Categorizes the text according to the BERT model.
   *
   * @param text text
   * @return list of labels, in order, highest prediction score first
   */
  @SuppressFBWarnings(
      value = "RCN_REDUNDANT_NULLCHECK_WOULD_HAVE_BEEN_A_NPE",
      justification = "Not just a nullcheck")
  public ImmutableList<LabelAndScore> categorize(String text) {
    log.debug("Received: {}", text);
    Session.Runner runner = model.session().runner();

    Features features = featureExtractor.extractFeatures(text);

    Example example = Example.newBuilder().setFeatures(features).build();

    byte[][] exampleBytes = {example.toByteArray()};

    try (Tensor<String> inputTensor = Tensors.create(exampleBytes)) {
      List<Tensor<?>> outputTensors =
          runner.feed(INPUT_TENSOR_NAME, inputTensor).fetch(OUTPUT_TENSOR_NAME).run();
      verify(
          outputTensors.size() == 1, "outputTensors.size() = %s, expected 1", outputTensors.size());

      try (Tensor<?> outputTensor = outputTensors.get(0)) {
        float[][] template = new float[1][model.labels().size()];
        float[][] copied = outputTensor.copyTo(template);

        return IntStream.range(0, copied[0].length)
            .mapToObj(i -> new LabelAndScore(model.labels().get(i), copied[0][i]))
            .sorted(Comparator.reverseOrder())
            .collect(toImmutableList());
      }
    }
  }

  /** Represents a label and its score. Immutable. */
  public static final class LabelAndScore implements Comparable<LabelAndScore> {
    private final String label;

    private final double predictionScore;

    private LabelAndScore(String label, double predictionScore) {
      this.label = checkNotNull(label);
      checkArgument(
          Range.closed(0d, 1d).contains(predictionScore),
          "Expected prediction score between 0 and 1.");
      this.predictionScore = predictionScore;
    }

    @Override
    public int compareTo(LabelAndScore that) {
      return Double.compare(this.predictionScore, that.predictionScore);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      LabelAndScore that = (LabelAndScore) o;
      return Double.compare(that.predictionScore, predictionScore) == 0
          && Objects.equals(label, that.label);
    }

    @Override
    public int hashCode() {
      return Objects.hash(label, predictionScore);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper("")
          .add("label", label)
          .add("predictionScore", predictionScore)
          .toString();
    }

    public String label() {
      return label;
    }

    public double predictionScore() {
      return predictionScore;
    }
  }
}
