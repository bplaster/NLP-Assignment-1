package edu.berkeley.nlp.assignments;

import edu.berkeley.nlp.io.PennTreebankReader;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;
import edu.berkeley.nlp.util.*;

import java.util.*;

/**
 * @author Dan Klein
 */
public class POSTaggerTester {

  static final String START_WORD = "<S>";
  static final String STOP_WORD = "</S>";
  static final String START_TAG = "<S>";
  static final String STOP_TAG = "</S>";

  /**
   * Tagged sentences are a bundling of a list of words and a list of their
   * tags.
   */
  static class TaggedSentence {
    List<String> words;
    List<String> tags;

    public int size() {
      return words.size();
    }

    public List<String> getWords() {
      return words;
    }

    public List<String> getTags() {
      return tags;
    }

    public String toString() {
      StringBuilder sb = new StringBuilder();
      for (int position = 0; position < words.size(); position++) {
        String word = words.get(position);
        String tag = tags.get(position);
        sb.append(word);
        sb.append("_");
        sb.append(tag);
      }
      return sb.toString();
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof TaggedSentence)) return false;

      final TaggedSentence taggedSentence = (TaggedSentence) o;

      if (tags != null ? !tags.equals(taggedSentence.tags) : taggedSentence.tags != null) return false;
      if (words != null ? !words.equals(taggedSentence.words) : taggedSentence.words != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = (words != null ? words.hashCode() : 0);
      result = 29 * result + (tags != null ? tags.hashCode() : 0);
      return result;
    }

    public TaggedSentence(List<String> words, List<String> tags) {
      this.words = words;
      this.tags = tags;
    }
  }

  /**
   * States are pairs of tags along with a position index, representing the two
   * tags preceding that position.  So, the START state, which can be gotten by
   * State.getStartState() is [START, START, 0].  To build an arbitrary state,
   * for example [DT, NN, 2], use the static factory method
   * State.buildState("DT", "NN", 2).  There isnt' a single final state, since
   * sentences lengths vary, so State.getEndState(i) takes a parameter for the
   * length of the sentence.
   */
  static class State {

    private static transient Interner<State> stateInterner = new Interner<State>(new Interner.CanonicalFactory<State>() {
      public State build(State state) {
        return new State(state);
      }
    });

    private static transient State tempState = new State();

    public static State getStartState() {
      return buildState(START_TAG, START_TAG, 0);
    }

    public static State getStopState(int position) {
      return buildState(STOP_TAG, STOP_TAG, position);
    }

    public static State buildState(String previousPreviousTag, String previousTag, int position) {
      tempState.setState(previousPreviousTag, previousTag, position);
      return stateInterner.intern(tempState);
    }

    public static List<String> toTagList(List<State> states) {
      List<String> tags = new ArrayList<String>();
      if (states.size() > 0) {
        tags.add(states.get(0).getPreviousPreviousTag());
        for (State state : states) {
          tags.add(state.getPreviousTag());
        }
      }
      return tags;
    }

    public int getPosition() {
      return position;
    }

    public String getPreviousTag() {
      return previousTag;
    }

    public String getPreviousPreviousTag() {
      return previousPreviousTag;
    }

    public State getNextState(String tag) {
      return State.buildState(getPreviousTag(), tag, getPosition() + 1);
    }

    public State getPreviousState(String tag) {
      return State.buildState(tag, getPreviousPreviousTag(), getPosition() - 1);
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof State)) return false;

      final State state = (State) o;

      if (position != state.position) return false;
      if (previousPreviousTag != null ? !previousPreviousTag.equals(state.previousPreviousTag) : state.previousPreviousTag != null) return false;
      if (previousTag != null ? !previousTag.equals(state.previousTag) : state.previousTag != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = position;
      result = 29 * result + (previousTag != null ? previousTag.hashCode() : 0);
      result = 29 * result + (previousPreviousTag != null ? previousPreviousTag.hashCode() : 0);
      return result;
    }

    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getPosition() + "]";
    }

    int position;
    String previousTag;
    String previousPreviousTag;

    private void setState(String previousPreviousTag, String previousTag, int position) {
      this.previousPreviousTag = previousPreviousTag;
      this.previousTag = previousTag;
      this.position = position;
    }

    private State() {
    }

    private State(State state) {
      setState(state.getPreviousPreviousTag(), state.getPreviousTag(), state.getPosition());
    }
  }

  /**
   * A Trellis is a graph with a start state an an end state, along with
   * successor and predecessor functions.
   */
  static class Trellis <S> {
    S startState;
    S endState;
    int sentenceLength = 0;
    Counter<String> possibleStates;
    CounterMap<S, S> forwardTransitions;
    CounterMap<S, S> backwardTransitions;

    /**
     * Get the unique start state for this trellis.
     */
    public S getStartState() {
      return startState;
    }

    public void setStartState(S startState) {
      this.startState = startState;
    }
    
    public int getSentenceLength(){
    	return this.sentenceLength;
    }
    
    public Counter<String> getPossibleStates(){
    	return this.possibleStates;
    }

    public void setSentenceLength(int length){
    	this.sentenceLength = length;
    }
    
    public void setPossibleStates(Counter<String> states){
    	this.possibleStates = states;
    }
    
    /**
     * Get the unique end state for this trellis.
     */
    public S getEndState() {
      return endState;
    }

    public void setStopState(S endState) {
      this.endState = endState;
    }

    /**
     * For a given state, returns a counter over what states can be next in the
     * markov process, along with the cost of that transition.  Caution: a state
     * not in the counter is illegal, and should be considered to have cost
     * Double.NEGATIVE_INFINITY, but Counters score items they don't contain as
     * 0.
     */
    public Counter<S> getForwardTransitions(S state) {
      return forwardTransitions.getCounter(state);

    }


    /**
     * For a given state, returns a counter over what states can precede it in
     * the markov process, along with the cost of that transition.
     */
    public Counter<S> getBackwardTransitions(S state) {
      return backwardTransitions.getCounter(state);
    }

    public void setTransitionCount(S start, S end, double count) {
      forwardTransitions.setCount(start, end, count);
      backwardTransitions.setCount(end, start, count);
    }

    public Trellis() {
      forwardTransitions = new CounterMap<S, S>();
      backwardTransitions = new CounterMap<S, S>();
    }
  }

  /**
   * A TrellisDecoder takes a Trellis and returns a path through that trellis in
   * which the first item is trellis.getStartState(), the last is
   * trellis.getEndState(), and each pair of states is connected in the
   * trellis.
   */
  static interface TrellisDecoder <S> {
    List<S> getBestPath(Trellis<S> trellis);
  }

  static class GreedyDecoder <S> implements TrellisDecoder<S> {
    public List<S> getBestPath(Trellis<S> trellis) {
      List<S> states = new ArrayList<S>();
      S currentState = trellis.getStartState();
      states.add(currentState);
      while (!currentState.equals(trellis.getEndState())) {
        Counter<S> transitions = trellis.getForwardTransitions(currentState);
        S nextState = transitions.argMax();
        states.add(nextState);
        currentState = nextState;
      }
      return states;
    }
  }
  
  static class ViterbiDecoder <S> implements TrellisDecoder<S> {
	  
	  public List<S> getBestPath(Trellis<S> trellis) {
		  
		  ArrayList<String> states = new ArrayList<String>();
		  states.addAll(trellis.getPossibleStates().keySet());
		  int start = 0;
		  if(!states.contains(START_TAG)){
			  states.add(start, START_TAG);
		  }
		  
		  int N = states.size();
		  int T = trellis.getSentenceLength() + 2;
		  double pi[][] = new double[T][N];
		  int[][] bp = new int[T][N];

		  // Initialize Values
		  for(int i = 0; i<T; i++){
			  for(int j = 0; j<N; j++){
				  bp[i][j] = start;
				  pi[i][j] = Double.NEGATIVE_INFINITY;
			  }
		  }
		  pi[0][start] = 0.0;

		  for(int i = 1; i<T; i++){
			  for(int j = 0; j<N; j++){
				  int b = bp[i][j];
				  double max = Double.NEGATIVE_INFINITY;

				  for(int k = 0; k<N; k++){
					  double p = pi[i-1][k];
					  S currentState = (S)State.buildState(states.get(k), states.get(j), i);
					  S previousState = (S)State.buildState(states.get(bp[i-1][k]), states.get(k), i-1);

					  double transition = Double.NEGATIVE_INFINITY;
					  if(trellis.getForwardTransitions(previousState).containsKey(currentState)){
						  transition = trellis.getForwardTransitions(previousState).getCount(currentState);
					  }
					  
					  double score = transition + p;
					  if(score > max){
						  max = score;
						  b = k;
					  }
				  }
				  pi[i][j] = max;
				  bp[i][j] = b;
			  }
		  }
		  
		  // Find Max Back Pointer
		  int currentBP = 0;
		  double m = Double.NEGATIVE_INFINITY;
		  for(int j = 0; j<N; j++){
			  double p = pi[T-1][j];
			  if(p > m){
				  m = p;
				  currentBP = j;
			  }
		  }
		  
		  // Determine Max Path
		  List<S> backwardPath = new ArrayList<S>();
		  S statePtr = trellis.getEndState();
		  backwardPath.add(statePtr);
		  for(int i = 1; i<=T; i++){
			  S st = (S)State.buildState(states.get(bp[T-i][currentBP]), states.get(currentBP), T-i);
			  currentBP = bp[T-i][currentBP];
			  backwardPath.add(st);
		  }

		  // Reverse list
		  List<S> path = new ArrayList<S>();
		  for(int i = 1; i<=backwardPath.size(); i++){
			  path.add(backwardPath.get(backwardPath.size() - i));
		  }

//		  System.out.println(path);

		  return path;
	  }
  }

  static class POSTagger {

    LocalTrigramScorer localTrigramScorer;
    TrellisDecoder<State> trellisDecoder;

    // chop up the training instances into local contexts and pass them on to the local scorer.
    public void train(List<TaggedSentence> taggedSentences) {
      localTrigramScorer.train(extractLabeledLocalTrigramContexts(taggedSentences));
    }

    // chop up the validation instances into local contexts and pass them on to the local scorer.
    public void validate(List<TaggedSentence> taggedSentences) {
      localTrigramScorer.validate(extractLabeledLocalTrigramContexts(taggedSentences));
    }

    private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(List<TaggedSentence> taggedSentences) {
      List<LabeledLocalTrigramContext> localTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
      for (TaggedSentence taggedSentence : taggedSentences) {
        localTrigramContexts.addAll(extractLabeledLocalTrigramContexts(taggedSentence));
      }
      return localTrigramContexts;
    }

    private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(TaggedSentence taggedSentence) {
      List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
      List<String> words = new BoundedList<String>(taggedSentence.getWords(), START_WORD, STOP_WORD);
      List<String> tags = new BoundedList<String>(taggedSentence.getTags(), START_TAG, STOP_TAG);
      for (int position = 0; position <= taggedSentence.size() + 1; position++) {
        labeledLocalTrigramContexts.add(new LabeledLocalTrigramContext(words, position, tags.get(position - 2), tags.get(position - 1), tags.get(position)));
      }
      return labeledLocalTrigramContexts;
    }

    /**
     * Builds a Trellis over a sentence, by starting at the state State, and
     * advancing through all legal extensions of each state already in the
     * trellis.  You should not have to modify this code (or even read it,
     * really).
     */
    private Trellis<State> buildTrellis(List<String> sentence) {
      Trellis<State> trellis = new Trellis<State>();
      trellis.setStartState(State.getStartState());
      trellis.setSentenceLength(sentence.size());
      trellis.setPossibleStates(localTrigramScorer.getTags());
      State stopState = State.getStopState(sentence.size() + 2);
      trellis.setStopState(stopState);
      Set<State> states = Collections.singleton(State.getStartState());
      for (int position = 0; position <= sentence.size() + 1; position++) {
        Set<State> nextStates = new HashSet<State>();
        for (State state : states) {
          if (state.equals(stopState))
            continue;
          LocalTrigramContext localTrigramContext = new LocalTrigramContext(sentence, position, state.getPreviousPreviousTag(), state.getPreviousTag());
          Counter<String> tagScores = localTrigramScorer.getLogScoreCounter(localTrigramContext);
          for (String tag : tagScores.keySet()) {
            double score = tagScores.getCount(tag);
            State nextState = state.getNextState(tag);
            trellis.setTransitionCount(state, nextState, score);
            nextStates.add(nextState);
          }
        }
//        System.out.println("States: "+nextStates);
        states = nextStates;
      }
      return trellis;
    }

    // to tag a sentence: build its trellis and find a path through that trellis
    public List<String> tag(List<String> sentence) {
      Trellis<State> trellis = buildTrellis(sentence);
      List<State> states = trellisDecoder.getBestPath(trellis);
      List<String> tags = State.toTagList(states);
      tags = stripBoundaryTags(tags);
      return tags;
    }

    /**
     * Scores a tagging for a sentence.  Note that a tag sequence not accepted
     * by the markov process should receive a log score of
     * Double.NEGATIVE_INFINITY.
     */
    public double scoreTagging(TaggedSentence taggedSentence) {
      double logScore = 0.0;
      List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = extractLabeledLocalTrigramContexts(taggedSentence);
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
        Counter<String> logScoreCounter = localTrigramScorer.getLogScoreCounter(labeledLocalTrigramContext);
        String currentTag = labeledLocalTrigramContext.getCurrentTag();
        if (logScoreCounter.containsKey(currentTag)) {
          logScore += logScoreCounter.getCount(currentTag);
        } else {
          logScore += Double.NEGATIVE_INFINITY;
        }
      }
      return logScore;
    }

    private List<String> stripBoundaryTags(List<String> tags) {
      return tags.subList(2, tags.size() - 2);
    }

    public POSTagger(LocalTrigramScorer localTrigramScorer, TrellisDecoder<State> trellisDecoder) {
      this.localTrigramScorer = localTrigramScorer;
      this.trellisDecoder = trellisDecoder;
    }
  }

  /**
   * A LocalTrigramContext is a position in a sentence, along with the previous
   * two tags -- basically a FeatureVector.
   */
  static class LocalTrigramContext {
    List<String> words;
    int position;
    String previousTag;
    String previousPreviousTag;

    public List<String> getWords() {
      return words;
    }

    public String getCurrentWord() {
      return words.get(position);
    }

    public int getPosition() {
      return position;
    }

    public String getPreviousTag() {
      return previousTag;
    }

    public String getPreviousPreviousTag() {
      return previousPreviousTag;
    }

    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "]";
    }

    public LocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag) {
      this.words = words;
      this.position = position;
      this.previousTag = previousTag;
      this.previousPreviousTag = previousPreviousTag;
    }
  }

  /**
   * A LabeledLocalTrigramContext is a context plus the correct tag for that
   * position -- basically a LabeledFeatureVector
   */
  static class LabeledLocalTrigramContext extends LocalTrigramContext {
    String currentTag;

    public String getCurrentTag() {
      return currentTag;
    }

    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "_" + getCurrentTag() + "]";
    }

    public LabeledLocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag, String currentTag) {
      super(words, position, previousPreviousTag, previousTag);
      this.currentTag = currentTag;
    }
  }

  /**
   * LocalTrigramScorers assign scores to tags occurring in specific
   * LocalTrigramContexts.
   */
  static interface LocalTrigramScorer {
    /**
     * The Counter returned should contain log probabilities, meaning if all
     * values are exponentiated and summed, they should sum to one (if it's a 
	 * single conditional probability). For efficiency, the Counter can 
	 * contain only the tags which occur in the given context 
	 * with non-zero model probability.
     */
	Counter<String> getTags();
	  
    Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext);

    void train(List<LabeledLocalTrigramContext> localTrigramContexts);

    void validate(List<LabeledLocalTrigramContext> localTrigramContexts);
  }

  
  /*
   * 
   */
  static class HMMTagScorer implements LocalTrigramScorer {
	    
	  boolean restrictTrigrams; // if true, assign log score of Double.NEGATIVE_INFINITY to illegal tag trigrams.
	    
	  double weights[] = {0.0,0.0,0.0};
	  CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
	  CounterMap<String, String> lowSuffixesToTags = new CounterMap<String, String>();
	  CounterMap<String, String> capSuffixesToTags = new CounterMap<String, String>();

	  Counter<String> unknownWordTags = new Counter<String>();
	  Counter<String> wordCount = new Counter<String>();
	  Counter<String> trigramsCount = new Counter<String>();
	  Counter<String> bigramsCount = new Counter<String>();
	  Counter<String> unigramsCount = new Counter<String>();
	  
	  // Tunable parameters
	  int suffixMaxLength = 10;
	  double smoothCount = 0.1;

	  public Counter<String> getTags() {
		  return unknownWordTags;
	  }
	  
	  public int getHistorySize() {
		  return 2;
	  }
	    
	  public Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) {
		  int position = localTrigramContext.getPosition();
		  String word = localTrigramContext.getWords().get(position);
		  Counter<String> tagCounter = unknownWordTags;
		  double p_of_w = 0.0;
		  
		  // Check if known word
		  if (wordsToTags.keySet().contains(word)) {
			  tagCounter = wordsToTags.getCounter(word);
			  p_of_w = wordCount.getCount(word);
		  } else {
			  tagCounter = getTagsForUnknownWord(word);
			  p_of_w = 1.0/wordCount.size();
		  }

		  String bigram = makeBigramString(localTrigramContext.getPreviousPreviousTag(),localTrigramContext.getPreviousTag());
		  double bCount = bigramsCount.getCount(bigram);
		  double uCount = unigramsCount.getCount(localTrigramContext.getPreviousTag());
		  double uTotCount = unigramsCount.totalCount();
		  Set<String> allowedFollowingTags = allowedFollowingTags(tagCounter.keySet(), localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag());
		  Counter<String> logScoreCounter = new Counter<String>();
		  
		  for (String tag : tagCounter.keySet()) {
			  String trigram = makeTrigramString(localTrigramContext.getPreviousPreviousTag(),localTrigramContext.getPreviousTag(), tag);
			  double p_of_t = unigramsCount.getCount(tag)/uTotCount;
			  // Interpolate
			  double p_w_given_t = tagCounter.getCount(tag)*p_of_w/p_of_t;
			  double p_t1_given_t2t3 = (weights[0]*p_of_t);
			  if (uCount > 0) { p_t1_given_t2t3 += (weights[1]*bigramsCount.getCount(makeBigramString(localTrigramContext.getPreviousTag(), tag))/uCount); }
			  if (bCount > 0) { p_t1_given_t2t3 += (weights[2]*trigramsCount.getCount(trigram)/bCount); }
			  double logScore = Math.log(p_t1_given_t2t3) + Math.log(p_w_given_t);
			  
			  if (!restrictTrigrams || allowedFollowingTags.isEmpty() || allowedFollowingTags.contains(tag))
				  logScoreCounter.setCount(tag, logScore);
		  }
		  
		  return logScoreCounter;
	  }
	  
	  private Counter<String> getTagsForUnknownWord(String word){
		  int max = Math.min(suffixMaxLength, word.length());
		  int lastIndex = word.length() - 1;
		  Counter<String> averageSuffixTags = new Counter<String>();
		  
		  CounterMap<String, String> suffixesToTags = this.lowSuffixesToTags;
		  if(Character.isUpperCase(word.charAt(0))){
			  suffixesToTags = this.capSuffixesToTags;
		  }
		  
		  // Average the tags over all the possible suffixes
		  for(int i = 0; i < max; i++){
			  Counter<String> suffixTags = suffixesToTags.getCounter(word.substring(lastIndex - i, lastIndex));
			  for(String tag: suffixTags.keySet()){
				  averageSuffixTags.incrementCount(tag, suffixTags.getCount(tag)/max);
			  }
		  }
		  
		  if (averageSuffixTags.isEmpty()){
			  averageSuffixTags = unknownWordTags;
		  }
		  averageSuffixTags = Counters.normalize(averageSuffixTags);
		  
		  return averageSuffixTags;
	  }

	  private Set<String> allowedFollowingTags(Set<String> tags, String previousPreviousTag, String previousTag) {
		  Set<String> allowedTags = new HashSet<String>();
		  for (String tag : tags) {
			  if (trigramsCount.containsKey(makeTrigramString(previousPreviousTag, previousTag, tag))) {
				  allowedTags.add(tag);
			  }
		  }
		  return allowedTags;
	  }
	  
	  private String makeTrigramString(String previousPreviousTag, String previousTag, String currentTag) {
		  return previousPreviousTag + " " + previousTag + " " + currentTag;
	  }
	  
	  private String makeBigramString(String previousPreviousTag, String previousTag) {
		  return previousPreviousTag + " " + previousTag;
	  }
	  
	  private void incrementSuffixes(String word, String tag, double count){
		  int max = Math.min(suffixMaxLength, word.length());
		  int lastIndex = word.length() - 1;
		  
		  CounterMap<String, String> suffixesToTags = this.lowSuffixesToTags;
		  if(Character.isUpperCase(word.charAt(0))){
			  suffixesToTags = this.capSuffixesToTags;
		  }
		  
		  for(int i = 0; i < max; i++){
			  suffixesToTags.incrementCount(word.substring(lastIndex - i, lastIndex), tag, count);
		  }
	  }

	  public void train(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
		  // collect word-tag counts
		  for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
			  String word = labeledLocalTrigramContext.getCurrentWord();
			  String tag = labeledLocalTrigramContext.getCurrentTag();
			  if (!wordsToTags.keySet().contains(word)) {
				  // word is currently unknown, so tally its tag in the unknown tag counter
				  unknownWordTags.incrementCount(tag, 1.0);
			  }
			  wordsToTags.incrementCount(word, tag, 1.0);
			  wordCount.incrementCount(word, 1.0);
			  incrementSuffixes(word, tag, 1.0);
			  unigramsCount.incrementCount(tag, 1.0);
			  bigramsCount.incrementCount(makeBigramString(labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag()), 1.0);
			  trigramsCount.incrementCount(makeTrigramString(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag()), 1.0);
		  }
		  
		  // Smoothing
		  for (String tag1: unigramsCount.keySet()){
			  unigramsCount.incrementCount(tag1, smoothCount);
			  for (String tag2: unigramsCount.keySet()){
				  // Smoothing for bigrams
				  String bigram = makeBigramString(tag1, tag2);
				  bigramsCount.incrementCount(bigram, smoothCount);
				  for (String tag3: unigramsCount.keySet()){
					  // Smoothing for trigrams
					  String trigram = makeTrigramString(tag1, tag2, tag3);
					  trigramsCount.incrementCount(trigram, smoothCount);
				  }
			  }
		  }
		  		  
		  // Determine weights for Interpolation
		  // http://www.coli.uni-saarland.de/~thorsten/publications/Brants-ANLP00.pdf
		  for (String trigram: trigramsCount.keySet()){
			  double f012 = trigramsCount.getCount(trigram);
			  String tags[] = trigram.split(" ");
			  String b01 = makeBigramString(tags[0], tags[1]);
			  String b12 = makeBigramString(tags[1], tags[2]);
			  double a = (f012-1.0)/(bigramsCount.getCount(b01)-1.0);
			  double b = (bigramsCount.getCount(b12)-1.0)/(unigramsCount.getCount(tags[1])-1.0);
			  double c = (unigramsCount.getCount(tags[2])-1.0)/(wordsToTags.totalCount()-1.0);

			  if(a > b && a > c){
				  weights [2] += f012; 
			  } else if (b > c) {
				  weights [1] += f012; 
			  } else {
				  weights [0] += f012; 
			  }
		  }
		  double len = weights[0] + weights[1] + weights[2];
		  for(int i=0; i<weights.length; i++){ weights[i] /= len; }
		  System.out.println("Interpolation weights: " + weights[0] + " " + weights[1] + " " + weights[2]);
		  
		  // Normalize
		  wordCount = Counters.normalize(wordCount);
		  wordsToTags = Counters.conditionalNormalize(wordsToTags);
		  lowSuffixesToTags = Counters.conditionalNormalize(lowSuffixesToTags);
		  capSuffixesToTags = Counters.conditionalNormalize(capSuffixesToTags);
		  unknownWordTags = Counters.normalize(unknownWordTags);
	  }

	  public void validate(List<LabeledLocalTrigramContext> localTrigramContexts) {
	      // no tuning for this dummy model!
	  }
	  
	  public HMMTagScorer(boolean restrictTrigrams) {
		  this.restrictTrigrams = restrictTrigrams;
	  }
  }
  
  /**
   * The MostFrequentTagScorer gives each test word the tag it was seen with
   * most often in training (or the tag with the most seen word types if the
   * test word is unseen in training.  This scorer actually does a little more
   * than its name claims -- if constructed with restrictTrigrams = true, it
   * will forbid illegal tag trigrams, otherwise it makes no use of tag history
   * information whatsoever.
   */
  static class MostFrequentTagScorer implements LocalTrigramScorer {

    boolean restrictTrigrams; // if true, assign log score of Double.NEGATIVE_INFINITY to illegal tag trigrams.

    CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
    Counter<String> unknownWordTags = new Counter<String>();
    Set<String> seenTagTrigrams = new HashSet<String>();

    public int getHistorySize() {
      return 2;
    }

    public Counter<String> getTags() {
    	return unknownWordTags;
    }
	  
    public Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) {
      int position = localTrigramContext.getPosition();
      String word = localTrigramContext.getWords().get(position);
      Counter<String> tagCounter = unknownWordTags;
      if (wordsToTags.keySet().contains(word)) {
        tagCounter = wordsToTags.getCounter(word);
      }
      Set<String> allowedFollowingTags = allowedFollowingTags(tagCounter.keySet(), localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag());
      Counter<String> logScoreCounter = new Counter<String>();
      for (String tag : tagCounter.keySet()) {
        double logScore = Math.log(tagCounter.getCount(tag));
        if (!restrictTrigrams || allowedFollowingTags.isEmpty() || allowedFollowingTags.contains(tag))
          logScoreCounter.setCount(tag, logScore);
      }
      return logScoreCounter;
    }

    private Set<String> allowedFollowingTags(Set<String> tags, String previousPreviousTag, String previousTag) {
      Set<String> allowedTags = new HashSet<String>();
      for (String tag : tags) {
        String trigramString = makeTrigramString(previousPreviousTag, previousTag, tag);
        if (seenTagTrigrams.contains((trigramString))) {
          allowedTags.add(tag);
        }
      }
      return allowedTags;
    }

    private String makeTrigramString(String previousPreviousTag, String previousTag, String currentTag) {
      return previousPreviousTag + " " + previousTag + " " + currentTag;
    }

    public void train(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
      // collect word-tag counts
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
        String word = labeledLocalTrigramContext.getCurrentWord();
        String tag = labeledLocalTrigramContext.getCurrentTag();
        if (!wordsToTags.keySet().contains(word)) {
          // word is currently unknown, so tally its tag in the unknown tag counter
          unknownWordTags.incrementCount(tag, 1.0);
        }
        wordsToTags.incrementCount(word, tag, 1.0);
        seenTagTrigrams.add(makeTrigramString(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag()));
      }
      wordsToTags = Counters.conditionalNormalize(wordsToTags);
      unknownWordTags = Counters.normalize(unknownWordTags);
    }

    public void validate(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
      // no tuning for this dummy model!
    }

    public MostFrequentTagScorer(boolean restrictTrigrams) {
      this.restrictTrigrams = restrictTrigrams;
    }
  }

  private static List<TaggedSentence> readTaggedSentences(String path, int low, int high) {
    Collection<Tree<String>> trees = PennTreebankReader.readTrees(path, low, high);
    List<TaggedSentence> taggedSentences = new ArrayList<TaggedSentence>();
    Trees.TreeTransformer<String> treeTransformer = new Trees.EmptyNodeStripper();
    for (Tree<String> tree : trees) {
      tree = treeTransformer.transformTree(tree);
      List<String> words = new BoundedList<String>(new ArrayList<String>(tree.getYield()), START_WORD, STOP_WORD);
      List<String> tags = new BoundedList<String>(new ArrayList<String>(tree.getPreTerminalYield()), START_TAG, STOP_TAG);
      taggedSentences.add(new TaggedSentence(words, tags));
    }
    return taggedSentences;
  }

  private static void evaluateTagger(POSTagger posTagger, List<TaggedSentence> taggedSentences, Set<String> trainingVocabulary, boolean verbose) {
    double numTags = 0.0;
    double numTagsCorrect = 0.0;
    double numUnknownWords = 0.0;
    double numUnknownWordsCorrect = 0.0;
    int numDecodingInversions = 0;
    for (TaggedSentence taggedSentence : taggedSentences) {
      List<String> words = taggedSentence.getWords();
      List<String> goldTags = taggedSentence.getTags();
      List<String> guessedTags = posTagger.tag(words);
      for (int position = 0; position < words.size() - 1; position++) {
        String word = words.get(position);
        String goldTag = goldTags.get(position);
        String guessedTag = guessedTags.get(position);
        if (guessedTag.equals(goldTag))
          numTagsCorrect += 1.0;
        numTags += 1.0;
        if (!trainingVocabulary.contains(word)) {
          if (guessedTag.equals(goldTag))
            numUnknownWordsCorrect += 1.0;
          numUnknownWords += 1.0;
        }
      }
      double scoreOfGoldTagging = posTagger.scoreTagging(taggedSentence);
      double scoreOfGuessedTagging = posTagger.scoreTagging(new TaggedSentence(words, guessedTags));
      if (scoreOfGoldTagging > scoreOfGuessedTagging) {
        numDecodingInversions++;
//        System.out.println(alignedTaggings(words, goldTags, guessedTags, true) + "\n");
        if (verbose) System.out.println("WARNING: Decoder suboptimality detected.  Gold tagging has higher score than guessed tagging.");
      }
      if (verbose) System.out.println(alignedTaggings(words, goldTags, guessedTags, true) + "\n");
    }
    System.out.println("Tag Accuracy: " + (numTagsCorrect / numTags) + " (Unknown Accuracy: " + (numUnknownWordsCorrect / numUnknownWords) + ")  Decoder Suboptimalities Detected: " + numDecodingInversions);
  }

  // pretty-print a pair of taggings for a sentence, possibly suppressing the tags which correctly match
  private static String alignedTaggings(List<String> words, List<String> goldTags, List<String> guessedTags, boolean suppressCorrectTags) {
    StringBuilder goldSB = new StringBuilder("Gold Tags: ");
    StringBuilder guessedSB = new StringBuilder("Guessed Tags: ");
    StringBuilder wordSB = new StringBuilder("Words: ");
    for (int position = 0; position < words.size(); position++) {
      equalizeLengths(wordSB, goldSB, guessedSB);
      String word = words.get(position);
      String gold = goldTags.get(position);
      String guessed = guessedTags.get(position);
      wordSB.append(word);
      if (position < words.size() - 1)
        wordSB.append(' ');
      boolean correct = (gold.equals(guessed));
      if (correct && suppressCorrectTags)
        continue;
      guessedSB.append(guessed);
      goldSB.append(gold);
    }
    return goldSB + "\n" + guessedSB + "\n" + wordSB;
  }

  private static void equalizeLengths(StringBuilder sb1, StringBuilder sb2, StringBuilder sb3) {
    int maxLength = sb1.length();
    maxLength = Math.max(maxLength, sb2.length());
    maxLength = Math.max(maxLength, sb3.length());
    ensureLength(sb1, maxLength);
    ensureLength(sb2, maxLength);
    ensureLength(sb3, maxLength);
  }

  private static void ensureLength(StringBuilder sb, int length) {
    while (sb.length() < length) {
      sb.append(' ');
    }
  }

  private static Set<String> extractVocabulary(List<TaggedSentence> taggedSentences) {
    Set<String> vocabulary = new HashSet<String>();
    for (TaggedSentence taggedSentence : taggedSentences) {
      List<String> words = taggedSentence.getWords();
      vocabulary.addAll(words);
    }
    return vocabulary;
  }

  public static void main(String[] args) {
    // Parse command line flags and arguments
    Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);

    // Set up default parameters and settings
    String basePath = ".";
    boolean verbose = false;
    boolean useValidation = true;

    // Update defaults using command line specifications

    // The path to the assignment data
    if (argMap.containsKey("-path")) {
      basePath = argMap.get("-path");
    }
    System.out.println("Using base path: " + basePath);

    // Whether to use the validation or test set
    if (argMap.containsKey("-test")) {
      String testString = argMap.get("-test");
      if (testString.equalsIgnoreCase("test"))
        useValidation = false;
    }
    System.out.println("Testing on: " + (useValidation ? "validation" : "test"));

    // Whether or not to print the individual errors.
    if (argMap.containsKey("-verbose")) {
      verbose = true;
    }

    // Read in data
    System.out.print("Loading training sentences...");
    List<TaggedSentence> trainTaggedSentences = readTaggedSentences(basePath, 200, 2199);
    Set<String> trainingVocabulary = extractVocabulary(trainTaggedSentences);
    System.out.println("done.");
    System.out.print("Loading validation sentences...");
    List<TaggedSentence> validationTaggedSentences = readTaggedSentences(basePath, 2200, 2299);
    System.out.println("done.");
    System.out.print("Loading test sentences...");
    List<TaggedSentence> testTaggedSentences = readTaggedSentences(basePath, 2300, 2399);
    System.out.println("done.");

    // Construct tagger components
    LocalTrigramScorer localTrigramScorer;
    TrellisDecoder<State> trellisDecoder;
    
    if (argMap.containsKey("-hmm")){
    	localTrigramScorer = new HMMTagScorer(false);
    } else {
    	localTrigramScorer = new MostFrequentTagScorer(false);
    }
    
    if (argMap.containsKey("-viterbi")){
    	trellisDecoder = new ViterbiDecoder<State>();
    } else {
    	trellisDecoder = new GreedyDecoder<State>();
    }

    // Train tagger
    POSTagger posTagger = new POSTagger(localTrigramScorer, trellisDecoder);
    posTagger.train(trainTaggedSentences);
    posTagger.validate(validationTaggedSentences);

    // Evaluation set, use either test of validation (for dev)
    final List<TaggedSentence> evalTaggedSentences;
    if (useValidation) {
    	evalTaggedSentences = validationTaggedSentences;
    } else {
    	evalTaggedSentences = testTaggedSentences;
    }
    
    // Test tagger
    evaluateTagger(posTagger, evalTaggedSentences, trainingVocabulary, verbose);
  }
}