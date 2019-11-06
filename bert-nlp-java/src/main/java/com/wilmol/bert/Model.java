package com.wilmol.bert;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Streams;
import java.io.File;
import java.nio.file.Files;
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

  private final Logger log = LogManager.getLogger();

  private final Session modelSession;

  private final int maxSeqLength;

  private final ImmutableMap<Integer, String> modelLabels;

  private final ImmutableMap<String, Long> modelVocab;

  /**
   * Constructor; loads the BERT model and associated files.
   *
   * @param modelPath path to the saved_model.pb file and variables folder
   * @param maxSeqLength {@code max_seq_length} value used to train the model
   * @param labelsPath path to the labels.txt file used to train the model
   * @param vocabPath path to the vocab.txt file used to train the model
   * @throws IllegalArgumentException if loading the model fails
   */
  public Model(String modelPath, int maxSeqLength, String labelsPath, String vocabPath) {
    log.traceEntry(
        "Model(modelPath={}, maxSeqLength={}, labelsPath={}, vocabPath={})",
        modelPath,
        maxSeqLength,
        labelsPath,
        vocabPath);
    log.debug("Loading model and creating session");
    try {
      // TODO(wilmol) use Resources#getResource here?
      SavedModelBundle model = SavedModelBundle.load(modelPath, "serve");
      modelSession = model.session();
    } catch (Exception e) {
      log.error("Failed to load model");
      throw new IllegalArgumentException(e);
    }

    this.maxSeqLength = maxSeqLength;

    log.debug("Loading labels.txt");
    try {
      // TODO(wilmol) use Resources#getResource here?
      modelLabels =
          Files.lines(new File(labelsPath).toPath())
              .map(line -> line.split(","))
              .collect(
                  toImmutableMap(split -> Integer.parseInt(split[0]), split -> split[1].trim()));
    } catch (Exception e) {
      log.error("Failed to read labels file", e);
      throw new IllegalArgumentException(e);
    }

    log.debug("Loading vocab.txt");
    try {
      // TODO(wilmol) use Resources#getResource here?
      modelVocab =
          Streams.mapWithIndex(Files.lines(new File(vocabPath).toPath()), Maps::immutableEntry)
              .collect(toImmutableMap(e -> e.getKey().trim(), Map.Entry::getValue));
    } catch (Exception e) {
      log.error("Failed to read vocab file", e);
      throw new IllegalArgumentException(e);
    }
  }

  /**
   * Closes the models session. The model will not be usable afterwards.
   *
   * @see Session#close
   */
  @Override
  public void close() {
    modelSession.close();
    log.info("Closed model session");
  }

  /**
   * Get the models session, session is thread safe.
   *
   * @return the models session
   */
  public Session session() {
    return modelSession;
  }

  /**
   * Get the models {@code max_seq_length} value.
   *
   * @return the models {@code max_seq_length} value
   */
  public int maxSeqLength() {
    return maxSeqLength;
  }

  /**
   * Get the loaded labels map.
   *
   * @return the loaded labels with id as key and label as value
   */
  public ImmutableMap<Integer, String> labels() {
    return modelLabels;
  }

  /**
   * Get the loaded vocab map.
   *
   * @return the loaded vocab with tokens as keys and indices as values
   */
  public ImmutableMap<String, Long> vocab() {
    return modelVocab;
  }
}
