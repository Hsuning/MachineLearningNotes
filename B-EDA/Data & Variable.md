# Data & Variable
```toc
```

## Unstructured Data
- Many sources, IoT (internet of things)
- Images: a collection of pixels, containing RGB colour information or Greyscale / intensity information
- Texts: sequences of words and nonword characters. Often organised by sections, subsections and so on.
- Clickstreams: sequences of actions by a user interacting with an app or a web page

## Structured Data
- Turn raw data into actionable information
- Table with rows and columns

### Two Basic Types
- Numeric and categorical
- Data type is important to help determine the type of visual display, data analysis, or statistical model
- Improve computational performance
- Determine how software will handle computations for that variables
- Storage and indexing can be optimized



#### Numerical (quantitative)
#NumericalVariable 
- Data that are expressed on a numeric scale

##### Discrete 
#DiscreteVariable  
- Data that can only take integer values, typically counts  
- Ex: number of visits, count of the occurrence of an event

##### Continuous 
#ContinuousVariable  
- Data that can take on any value in an interval (range)
- Ex: height in cm, pocket depth in mm, wind speed, time duration

#### Categorical (qualitative)
#CategoricalVariable #NorminalVariable  
- Data that can take on only a specific set of values representing a set of possible categories
- Ex: type of TV screen

##### Ordinal 
#OrdinalVariable  
- Data that has an explicit ordering  
- Ex: minimal/moderate/severe/unbearable pain; easy/medium/hard
- Can be represented with `sklearn.preprocessing.OrdinalEncoder`, preserving a user-specified ordering in visualisation

##### Binary
#BinaryVariable #Boolean
- With just two categories of values
- Ex: 0/1, yes/no, or true/false


##  Data frame  / Rectangular Data
#RectangularData #DataFrame
- rectangular data like spreadsheet, database table
- two-dimensional matrix with rows indicating records (cases) and columns indicating features (variables)
- unstructured data must be processed and manipulated so that it can be represented as a set of features in the rectangular data

### Feature, attribute, predictor, variable
- A column within a table

### Target variable, dependent variable, target, output
- What we want to predict

### Records, case, example, instance, pattern, sample
- A row within a table


## Nonrectangular Data
#NonRectangulardata
- Time series data #TimeSeries: records successive measurements of the same variable. ex: data produced by devices (IoT)
- Spatial data #SpatialData #GeoLocation: 
	- object view: object and its spatial coordinates
	- field view: focuses on small units of space and the value of a relevant metrix (pixel brightness)
- Graph data #GraphData: represent physical, social, and abstract relationship
	- graph of social network
	- distribution hubs connected by roads
	- useful for network optimization and recommender systems
