import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class NBayes
{
	static List<Map<Pair, Integer>> listOfMaps = new ArrayList<Map<Pair, Integer>>();	
	static List<Map<Pair, Double>> probList = new ArrayList<Map<Pair, Double>>();
	
	static HashMap<Integer, String> header = new HashMap<Integer, String>();	
	static HashMap<Integer, Integer> labelCount = new HashMap<Integer, Integer>();
	
	static ArrayList<Integer> label = new ArrayList<Integer>();
	static ArrayList<Integer> prediction = new ArrayList<Integer>();
	
	private static int instanceCount = 0;	
	
	//binary classification
	final private static int classCount = 2;
	
	private static HashMap<Integer, Double> priorProb = new HashMap<Integer, Double>();
	
	//Tuple of attribute and label
	public static class Pair
	{
		int attrValue;
		int lab;
		
		Pair(int v1, int v2)
		{
			attrValue = v1;
			lab = v2;
		}
		
		public boolean equals(Object o)
		{
			final Pair other = (Pair)o;
			if(other.attrValue == attrValue && other.lab == lab)
				return true;
			else
				return false;
		}		
		
		public int hashCode()
		{
			int hash = 3;
			hash = hash * 53 + attrValue + lab;
			return hash;
		}
		
		public void display()
		{
			System.out.print(attrValue + " " + lab);
		}
	}
	
	/* parse the data to build data structure */
	private static void parseTrainData(String train) throws FileNotFoundException, IOException
	{
		BufferedReader reader = new BufferedReader(new FileReader(train));		
		
		String line = reader.readLine();
		String[] words = line.split("\\s+");
		
		//First line is header
		for(int i=0;i<words.length-1;++i)
		{
			//First line is header store the attribute names
			header.put(i, words[i]);
			
			//initialize listOfMaps and probList with empty hash map otherwise it throw exception for the first access.
			Map<Pair, Integer> temp1 = new HashMap<Pair, Integer>();
			Map<Pair, Double> temp2 = new HashMap<Pair, Double>();
			listOfMaps.add(temp1);
			probList.add(temp2);
		}		
		
		while((line = reader.readLine()) != null)
		{
			words = line.split("\\s+");
			int lab = Integer.parseInt(words[words.length-1]);
			for(int i=0;i<words.length-1;++i)
			{
				int attr = Integer.parseInt(words[i]);
				
				Map<Pair, Integer> val = listOfMaps.get(i);
				Pair p = new Pair(attr, lab);
				
				//count the label's for each attribute
				if(val.containsKey(p))
				{
					listOfMaps.get(i).put(p, val.get(p)+1);
				}
				else
				{
					listOfMaps.get(i).put(p, 1);
				}
			}
			
			/* count total number of zeros and ones in the label column for computing 
			   one's probability and zero's probability in the data set */
			if(labelCount.containsKey(lab))
			{
				labelCount.put(lab, labelCount.get(lab)+1);
			}
			else
			{
				labelCount.put(lab, 1);
			}
			instanceCount++;
		}
		
		//Once entire data is parsed, compute the probability and build table which will be used during prediction.
		for(int i=0;i<listOfMaps.size();++i)
		{
			Map<Pair, Integer> val = listOfMaps.get(i);
			for(Pair p : val.keySet())
			{
				//total attribute value count divided by total label count of that value
				double prob = val.get(p) / (double)labelCount.get(p.lab);				
				
				probList.get(i).put(p, prob);
			}			
		}		
		reader.close();
	}
	
	private static int maxProbClass(double[] prob)
	{
		double val = Double.MIN_NORMAL;
		int index = -1;
		for(int i=0;i<prob.length;++i)
		{
			if(prob[i] > val)
			{
				val = prob[i];
				index = i;
			}
		}
		return index;
	}
	
	/* Make prediction for the data using the probability table */
	public static void predict(String data) throws IOException
	{
		BufferedReader reader = new BufferedReader(new FileReader(data));
		String line = "";
		
		//ignore header
		reader.readLine();		
		
		while((line = reader.readLine()) != null)
		{
			String[] words = line.split("\\s+");						
			
			double[] prob = new double[classCount];
		
			for(int clas=0;clas<classCount;++clas)
			{
				prob[clas] = priorProb.get(clas);
				for(int i=0;i<words.length-1;++i)
				{
					//compute the joint probability by multiplying conditional probability for all attributes(for both labels 0 and 1)
					Pair p = new Pair(Integer.parseInt(words[i]), clas);
				
					Map<Pair, Double> m = probList.get(i);				
					prob[clas] *= m.get(p);
				}
			}	
			
			//max probability label is the prediction
			prediction.add(maxProbClass(prob));
			
			int lab = Integer.parseInt(words[words.length-1]);
			//Store all label in the given order. This will be used will calculating accuracy(prediction vs actual)
			label.add(lab);	
			instanceCount++;
		}
		reader.close();
	}
	
	private static void displayLearnedParameters()
	{
		System.out.println("===============================================================");
		System.out.println("The learned parameters are :");
		for(int clas=0;clas<classCount;++clas)
		{
			String disp = "P(C=" + clas + ")=" + Double.toString(priorProb.get(clas)) + " ";
			for(int attr=0;attr<header.size();++attr)
			{
				for(int attrVal=0;attrVal<2;++attrVal)
				{
					Map<Pair, Double> m1 = probList.get(attr);
					Pair p = new Pair(attrVal, clas);
					String prob = Double.toString(m1.get(p));					
			
					disp += "P(" + header.get(attr) + "=" + attrVal + "|" + clas + ")=" + prob + " ";
				}
			}
			System.out.println(disp);
			System.out.println();
		}
		System.out.println("===============================================================");
	}
	
	/* compute accuracy for the prediction vs actual values */
	private static double calcAccuracy()
	{
		int errCount = 0;
		
		for(int i=0;i<instanceCount;++i)
		{
			int l = label.get(i);
			int p = prediction.get(i);
			if(Math.abs(l - p) > 0)
			{
				errCount++;
			}			
		}		
		//accuracy in percentage
		double acc = (double)(instanceCount - errCount)/instanceCount;
		return acc * 100;
	}
	
	private static void computePriorProb()
	{		
		for(int clas=0;clas<classCount;++clas)
		{
			double prob = (double)labelCount.get(clas) / instanceCount;
			priorProb.put(clas, prob);			
		}
		
		//instance count is computed in parseData() which will be used here. predict() also computes instance count since
		//test data also need to compute instance count. Hence reset to zero.
		//Note parseData is only called for training data. predict() will be called for both test and training
		//If we don't compute instance count in predict then test data will use training instance count which will be a problem
		instanceCount = 0;
	}
	
	public static void main(String[] args) throws IOException
	{
		//the program requires two command line arguments train.dat and test.dat in the same order
		String train = args[0];
		String test = args[1];
		
		parseTrainData(train);
		
		computePriorProb();
		
		predict(train);
		
		displayLearnedParameters();
		
		System.out.format("Accuracy on training set(%d instances) : %f\n", instanceCount, calcAccuracy());
		
		//reset common data structure which will be updated using test data for test prediction
		prediction.clear();
		label.clear();
		instanceCount = 0;
			
		predict(test);
		System.out.format("Accuracy on test set(%d instances) : %f\n", instanceCount, calcAccuracy());
	}	
}
