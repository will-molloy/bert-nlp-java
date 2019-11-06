package com.wilmol.bert;

import java.util.List;

/**
 * Tokenizer definition.
 *
 * @author wilmol
 */
public interface Tokenizer {

  /**
   * Tokenizes the given text.
   *
   * @param text text to tokenize.
   * @return list of tokens.
   */
  List<String> tokenize(String text);
}
