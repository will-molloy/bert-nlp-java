package com.wilmol.bert;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Streams;
import com.google.common.io.Resources;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;

/**
 * Represents a loaded BERT model.
 *
 * @author wilmol
 */
public class Model implements AutoCloseable {

  private static final String TAG_NAME = "serve";

  private final Logger log = LogManager.getLogger();

  private final Session modelSession;

  private final int maxSeqLength;

  private final ImmutableMap<String, Long> modelVocab;

  private final ImmutableMap<Integer, String> modelLabels;

  /**
   * Constructor.
   *
   * @param modelPath path to the saved_model.pb file and variables folder.
   * @param maxSeqLength {@code max_seq_length} value used to train the model.
   * @param vocabPath path to the vocab.txt file.
   * @param labelsPath path to the labels.txt file.
   */
  public Model(String modelPath, int maxSeqLength, String vocabPath, String labelsPath) {

    log.info("Loading model");
    try {
      SavedModelBundle model = SavedModelBundle.load(modelPath, TAG_NAME);
      modelSession = model.session();
    } catch (Exception e) {
      throw new IllegalArgumentException(
          String.format("Unable to load model from %s", modelPath), e);
    }

    checkArgument(maxSeqLength > 2);
    this.maxSeqLength = maxSeqLength;

    log.info("Loading vocab");
    try {
      modelVocab =
          Streams.mapWithIndex(
                  Resources.readLines(Resources.getResource(vocabPath), StandardCharsets.UTF_8)
                      .stream(),
                  Maps::immutableEntry)
              .collect(toImmutableMap(e -> e.getKey().trim(), Map.Entry::getValue));
    } catch (Exception e) {
      throw new IllegalArgumentException(
          String.format("Unable to load model vocab from %s", vocabPath), e);
    }

    log.info("Loading labels");
    try {
      modelLabels =
          Resources.readLines(Resources.getResource(labelsPath), StandardCharsets.UTF_8).stream()
              .map(line -> line.split(","))
              .collect(
                  toImmutableMap(split -> Integer.parseInt(split[0]), split -> split[1].trim()));
    } catch (Exception e) {
      throw new IllegalArgumentException(
          String.format("Unable to load model labels from %s", labelsPath), e);
    }
  }

  /**
   * The initialised model session. Threadsafe.
   *
   * @return The initialised model session. Threadsafe.
   */
  public Session session() {
    return modelSession;
  }

  /**
   * The {@code max_seq_length} value used to train the model.
   *
   * @return The {@code max_seq_length} value used to train the model.
   */
  public int maxSeqLength() {
    return maxSeqLength;
  }

  /**
   * The vocab used to train the model.
   *
   * @return the model vocab, loaded from vocab.txt.
   */
  public ImmutableMap<String, Long> vocab() {
    return modelVocab;
  }

  /**
   * The labels used to train the model.
   *
   * @return the model labels, loaded from labels.txt.
   */
  public ImmutableMap<Integer, String> labels() {
    return modelLabels;
  }

  /**
   * Closes the models session.
   *
   * @see Session#close
   */
  @Override
  public void close() {
    session().close();
  }
}
