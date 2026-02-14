import java.io.IOException;
import java.util.StringTokenizer;  
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {
  public static class TokenizerMapper
      extends Mapper<LongWritable, Text, Text, IntWritable> {
    
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    
    public void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException {
      
      // Remove punctuation using replaceAll()
      String line = value.toString();
      line = line.replaceAll("[^a-zA-Z\\s]", "");  // Remove all non-letter characters
      line = line.toLowerCase();  // Convert to lowercase
      
      // Use StringTokenizer to split into words
      StringTokenizer itr = new StringTokenizer(line);
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken());
        context.write(word, one);
      }
    }
  }
  
  public static class IntSumReducer
      extends Reducer<Text, IntWritable, Text, IntWritable> {

    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {

      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "word count");
    
    job.setJarByClass(WordCount.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    
    // QUESTION 9: Set split size if provided as third argument
    if (args.length > 2) {
        long splitSize = Long.parseLong(args[2]);
        job.getConfiguration().setLong(
            "mapreduce.input.fileinputformat.split.maxsize", 
            splitSize
        );
        System.out.println("==========================================");
        System.out.println("Split size set to: " + splitSize + " bytes");
        System.out.println("==========================================");
    } else {
        System.out.println("==========================================");
        System.out.println("Using default split size");
        System.out.println("==========================================");
    }
    
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    
    // QUESTION 9: Measure execution time
    System.out.println("Starting WordCount job...");
    long startTime = System.currentTimeMillis();
    
    boolean success = job.waitForCompletion(true);
    
    long endTime = System.currentTimeMillis();
    long executionTime = endTime - startTime;
    
    System.out.println("==========================================");
    System.out.println("Job completed: " + (success ? "SUCCESS" : "FAILED"));
    System.out.println("Total Execution Time: " + executionTime + " ms");
    System.out.println("Total Execution Time: " + (executionTime / 1000.0) + " seconds");
    System.out.println("==========================================");
    
    System.exit(success ? 0 : 1);
}
}
