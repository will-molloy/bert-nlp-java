package com.wilmol.bert;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Streams;
import java.io.File;
import java.nio.file.Files;
import java.util.Map;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;

/**
 * Represents a loaded BERT model and vocab.
 *
 * @author wilmol
 */
public class Model implements AutoCloseable {

  private static final String TAG_NAME = "serve";

  private final Session modelSession;

  private final ImmutableMap<String, Long> modelVocab;

  private final int maxSeqLength;

  /**
   * Constructor.
   *
   * @param modelPath path to the saved_model.pb file and variables folder.
   * @param vocabPath path to the vocab.txt file.
   * @param maxSeqLength {@code max_seq_length} value used to train the model.
   */
  public Model(String modelPath, String vocabPath, int maxSeqLength) {

    try {
      SavedModelBundle model = SavedModelBundle.load(modelPath, TAG_NAME);
      modelSession = model.session();
    } catch (Exception e) {
      throw new IllegalArgumentException(
          String.format("Unable to load model from %s", modelPath), e);
    }

    try {
      modelVocab =
          Streams.mapWithIndex(Files.lines(new File(vocabPath).toPath()), Maps::immutableEntry)
              .collect(toImmutableMap(e -> e.getKey().trim(), Map.Entry::getValue));
    } catch (Exception e) {
      throw new IllegalArgumentException(
          String.format("Unable to load model vocab from %s", vocabPath), e);
    }

    checkArgument(maxSeqLength > 2);
    this.maxSeqLength = maxSeqLength;
  }

  /**
   * The models session.
   *
   * @return The initialised model session. Threadsafe.
   */
  public Session session() {
    return modelSession;
  }

  /**
   * The models vocab.
   *
   * @return The model vocab, loaded from vocab.txt.
   */
  public ImmutableMap<String, Long> vocab() {
    return modelVocab;
  }

  /**
   * The models {@code max_seq_length}.
   *
   * @return The {@code max_seq_length} value used to train the model.
   */
  public int maxSeqLength() {
    return maxSeqLength;
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
