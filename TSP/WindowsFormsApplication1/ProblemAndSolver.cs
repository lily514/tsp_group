using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Diagnostics;
using System.Linq;

namespace TSP
{

    class ProblemAndSolver
    {

        private class TSPSolution
        {
            /// <summary>
            /// we use the representation [cityB,cityA,cityC] 
            /// to mean that cityB is the first city in the solution, cityA is the second, cityC is the third 
            /// and the edge from cityC to cityB is the final edge in the path.  
            /// You are, of course, free to use a different representation if it would be more convenient or efficient 
            /// for your data structure(s) and search algorithm. 
            /// </summary>
            public ArrayList
                Route;

            /// <summary>
            /// constructor
            /// </summary>
            /// <param name="iroute">a (hopefully) valid tour</param>
            public TSPSolution(ArrayList iroute)
            {
                Route = new ArrayList(iroute);
            }

            /// <summary>
            /// Compute the cost of the current route.  
            /// Note: This does not check that the route is complete.
            /// It assumes that the route passes from the last city back to the first city. 
            /// </summary>
            /// <returns></returns>
            public double costOfRoute()
            {
                // go through each edge in the route and add up the cost. 
                int x;
                City here;
                double cost = 0D;

                for (x = 0; x < Route.Count - 1; x++)
                {
                    here = Route[x] as City;
                    cost += here.costToGetTo(Route[x + 1] as City);
                }

                // go from the last city to the first. 
                here = Route[Route.Count - 1] as City;
                cost += here.costToGetTo(Route[0] as City);
                return cost;
            }
        }

        #region Private members 

        /// <summary>
        /// Default number of cities (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Problem Size text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int DEFAULT_SIZE = 25;

        /// <summary>
        /// Default time limit (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Time text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int TIME_LIMIT = 60;        //in seconds

        private const int CITY_ICON_SIZE = 5;


        // For normal and hard modes:
        // hard mode only
        private const double FRACTION_OF_PATHS_TO_REMOVE = 0.20;

        /// <summary>
        /// the cities in the current problem.
        /// </summary>
        private City[] Cities;
        /// <summary>
        /// a route through the current problem, useful as a temporary variable. 
        /// </summary>
        private ArrayList Route;
        /// <summary>
        /// best solution so far. 
        /// </summary>
        private TSPSolution bssf; 

        /// <summary>
        /// how to color various things. 
        /// </summary>
        private Brush cityBrushStartStyle;
        private Brush cityBrushStyle;
        private Pen routePenStyle;


        /// <summary>
        /// keep track of the seed value so that the same sequence of problems can be 
        /// regenerated next time the generator is run. 
        /// </summary>
        private int _seed;
        /// <summary>
        /// number of cities to include in a problem. 
        /// </summary>
        private int _size;

        /// <summary>
        /// Difficulty level
        /// </summary>
        private HardMode.Modes _mode;

        /// <summary>
        /// random number generator. 
        /// </summary>
        private Random rnd;

        /// <summary>
        /// time limit in milliseconds for state space search
        /// can be used by any solver method to truncate the search and return the BSSF
        /// </summary>
        private int time_limit;
        #endregion

        #region Public members

        /// <summary>
        /// These three constants are used for convenience/clarity in populating and accessing the results array that is passed back to the calling Form
        /// </summary>
        public const int COST = 0;           
        public const int TIME = 1;
        public const int COUNT = 2;
        
        public int Size
        {
            get { return _size; }
        }

        public int Seed
        {
            get { return _seed; }
        }
        #endregion

        #region Constructors
        public ProblemAndSolver()
        {
            this._seed = 1; 
            rnd = new Random(1);
            this._size = DEFAULT_SIZE;
            this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }

        public ProblemAndSolver(int seed)
        {
            this._seed = seed;
            rnd = new Random(seed);
            this._size = DEFAULT_SIZE;
            this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }

        public ProblemAndSolver(int seed, int size)
        {
            this._seed = seed;
            this._size = size;
            rnd = new Random(seed);
            this.time_limit = TIME_LIMIT * 1000;                        // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }
        public ProblemAndSolver(int seed, int size, int time)
        {
            this._seed = seed;
            this._size = size;
            rnd = new Random(seed);
            this.time_limit = time*1000;                        // time is entered in the GUI in seconds, but timer wants it in milliseconds

            this.resetData();
        }
        #endregion

        #region Private Methods

        /// <summary>
        /// Reset the problem instance.
        /// </summary>
        private void resetData()
        {

            Cities = new City[_size];
            Route = new ArrayList(_size);
            bssf = null;

            if (_mode == HardMode.Modes.Easy)
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble());
            }
            else // Medium and hard
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() * City.MAX_ELEVATION);
            }

            HardMode mm = new HardMode(this._mode, this.rnd, Cities);
            if (_mode == HardMode.Modes.Hard)
            {
                int edgesToRemove = (int)(_size * FRACTION_OF_PATHS_TO_REMOVE);
                mm.removePaths(edgesToRemove);
            }
            City.setModeManager(mm);

            cityBrushStyle = new SolidBrush(Color.Black);
            cityBrushStartStyle = new SolidBrush(Color.Red);
            routePenStyle = new Pen(Color.Blue,1);
            routePenStyle.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// make a new problem with the given size.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode)
        {
            this._size = size;
            this._mode = mode;
            resetData();
        }

        /// <summary>
        /// make a new problem with the given size, now including timelimit paremeter that was added to form.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode, int timelimit)
        {
            this._size = size;
            this._mode = mode;
            this.time_limit = timelimit*1000;                                   //convert seconds to milliseconds
            resetData();
        }

        /// <summary>
        /// return a copy of the cities in this problem. 
        /// </summary>
        /// <returns>array of cities</returns>
        public City[] GetCities()
        {
            City[] retCities = new City[Cities.Length];
            Array.Copy(Cities, retCities, Cities.Length);
            return retCities;
        }

        /// <summary>
        /// draw the cities in the problem.  if the bssf member is defined, then
        /// draw that too. 
        /// </summary>
        /// <param name="g">where to draw the stuff</param>
        public void Draw(Graphics g)
        {
            float width  = g.VisibleClipBounds.Width-45F;
            float height = g.VisibleClipBounds.Height-45F;
            Font labelFont = new Font("Arial", 10);

            // Draw lines
            if (bssf != null)
            {
                // make a list of points. 
                Point[] ps = new Point[bssf.Route.Count];
                int index = 0;
                foreach (City c in bssf.Route)
                {
                    if (index < bssf.Route.Count -1)
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[index+1]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    else 
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[0]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    ps[index++] = new Point((int)(c.X * width) + CITY_ICON_SIZE / 2, (int)(c.Y * height) + CITY_ICON_SIZE / 2);
                }

                if (ps.Length > 0)
                {
                    g.DrawLines(routePenStyle, ps);
                    g.FillEllipse(cityBrushStartStyle, (float)Cities[0].X * width - 1, (float)Cities[0].Y * height - 1, CITY_ICON_SIZE + 2, CITY_ICON_SIZE + 2);
                }

                // draw the last line. 
                g.DrawLine(routePenStyle, ps[0], ps[ps.Length - 1]);
            }

            // Draw city dots
            foreach (City c in Cities)
            {
                g.FillEllipse(cityBrushStyle, (float)c.X * width, (float)c.Y * height, CITY_ICON_SIZE, CITY_ICON_SIZE);
            }

        }

        /// <summary>
        ///  return the cost of the best solution so far. 
        /// </summary>
        /// <returns></returns>
        public double costOfBssf ()
        {
            if (bssf != null)
                return (bssf.costOfRoute());
            else
                return -1D; 
        }

        /// <summary>
        /// This is the entry point for the default solver
        /// which just finds a valid random tour 
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] defaultSolveProblem()
        {
            int i, swap, temp, count=0;
            string[] results = new string[3];
            int[] perm = new int[Cities.Length];
            Route = new ArrayList();
            Random rnd = new Random();
            Stopwatch timer = new Stopwatch();

            timer.Start();

            do
            {
                for (i = 0; i < perm.Length; i++)                                 // create a random permutation template
                    perm[i] = i;
                for (i = 0; i < perm.Length; i++)
                {
                    swap = i;
                    while (swap == i)
                        swap = rnd.Next(0, Cities.Length);
                    temp = perm[i];
                    perm[i] = perm[swap];
                    perm[swap] = temp;
                }
                Route.Clear();
                for (i = 0; i < Cities.Length; i++)                            // Now build the route using the random permutation 
                {
                    Route.Add(Cities[perm[i]]);
                }
                bssf = new TSPSolution(Route);
                count++;
            } while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found
            timer.Stop();

            results[COST] = costOfBssf().ToString();                          // load results array
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            return results;
        }

        /**
         * HeapQueue is a priority queue adapted from the networking lab
         * The priority queue contains bBNode objects and is arranged so that the parent
         * is closer to becoming a new bssf than its children.
         */
        class HeapQueue
        {
            // queue of branch and bound nodes, ordered by priority
            private bBNode[] queue;
            int count,states,maxcount = 0;

            /**
             * Returns the best node in the queue.
             * Replaces the top node with the last node, bubbles down, and decreases size of queue.
             */
            public bBNode DeleteBestRoute()
            {
                bBNode item = queue[1];
                BubbleDown(queue[count], 1);
                queue[count] = null;
                count--;
                return item;

            }

            /**
             * Inserts a node into the array at the bottom and lets it bubble up. 
             */
            public void Insert(bBNode item)
            {
                item.SetIndex(states);
                states++;
                count++;
                if (count > maxcount) maxcount = count;
                BubbleUp(item, count);
            }

            /**
             * Checks if the queue is empty 
             */
            public bool IsEmpty()
            {
                if (count == 0)
                    return true;
                else
                    return false;
            }

            /**
             * Initializes the queue with the size and the first node. 
             */
            public void MakeQueue(int size, bBNode start)
            {
                queue = new bBNode[size + 1];
                Insert(start);
            }


            /**
             * Lets a node move up the queue while it has a higher priority than its parent 
             */
            private void BubbleUp(bBNode item, int index)
            {
                while (index != 1 && queue[index/2] < item)
                {
                    queue[index] = queue[index / 2];
                    queue[index / 2].SetIndex(index);
                    index = index / 2;

                }
                queue[index] = item;
                item.SetIndex(index);
            }

            /**
             * Lets a node move down the queue while it has a lower priority than its children 
             */
            private void BubbleDown(bBNode item, int index)
            {
                int child_index = MaxChild(index);
                bBNode child_item = queue[child_index];
                while (child_index != 0 && child_item > item)
                {
                    queue[index] = child_item;
                    child_item.SetIndex(index);
                    index = child_index;
                    child_index = MaxChild(index);
                    child_item = queue[child_index];
                }
                queue[index] = item;
                item.SetIndex(index);
            }

            /**
             * Given an index into the queue, this returns the child with the higher priority 
             */
            private int MaxChild(int index)
            {
                if (2 * index > count)
                {
                    return 0;
                }
                else
                {
                    bBNode rh_child = queue[2 * index];
                    bBNode lh_child = queue[2 * index + 1];
                    if (rh_child > lh_child)
                        return 2*index;
                    else
                        return 2*index + 1;
                }
            }

            //returns the maximum stored states at one time
            public int getMaxStoredStates()
            {
                return this.maxcount;
            }

            //returns size of the queue
            public int getCount()
            {
                return this.count;
            }
        }

        /**
         * bBNode (branch and Bound Node)
         * Object contains a reduction matrix, the bound value, and a partial route list among other helper objects.
         */
        private class bBNode
        {
            //reduction matrix
            private double[,] matrix;

            //keeps a partial route (list of cities) in the order they are added
            private ArrayList route;

            //
            private double bound;

            //reverse lookup into the cities array
            private ArrayList lookup;

            //keeps track of whether city[i] has been used already
            private bool[] used;
            
            //index in priority queue
            private int index;

            //reference to list of cities needed for calculations
            private City[] cities;


            //creates a starting point from a list of cities
            public bBNode(City[] cities) {

                matrix = new double[cities.Length, cities.Length];
                used = new bool[cities.Length];
                for (int i = 0; i < cities.Length; i++)
                {
                    used[i] = false;
                    for (int j = 0; j < cities.Length; j++)
                    {
                        if (i == j)
                        {
                            matrix[i, j] = Double.PositiveInfinity;
                        }
                        else
                        {
                            matrix[i, j] = cities[i].costToGetTo(cities[j]);
                        }
                    }
                }
                this.bound = 0;
                this.route = new ArrayList();
                this.lookup = new ArrayList();
                this.index = 0;
                this.cities = cities;
            }

            //creates a new node given its parent and the index of which city it should follow next
            public bBNode(bBNode parent, int chosen)
            {
                //keep a reference to the list of cities
                this.cities = parent.cities;

                //make deep copies of important data structures
                this.route = new ArrayList(parent.route.ToArray());
                this.lookup = new ArrayList(parent.lookup.ToArray());
                this.used = new bool[cities.Length];
                parent.used.CopyTo(this.used, 0);
                this.matrix = parent.matrix.Clone() as double[,];
                this.bound = 0;
                bound += parent.bound;
                               
                
                //add the next chosen city to the route
                route.Add(cities[chosen]);
                //mark the city as used
                used[chosen] = true;
                //add the index of the city to the lookup array
                lookup.Add(chosen);

                //handle & reduce matrix
                if (route.Count > 1)
                {
                    int prev = (int)lookup[lookup.Count - 2];
                    this.bound += matrix[prev, chosen];
                    matrix[prev, chosen] = double.PositiveInfinity;
                    matrix[chosen, prev] = double.PositiveInfinity;
                    infiniteRow(prev);
                    infiniteCol(chosen);
                }
                Reduce();
            }

            //get and set the index in the priority queue
            public void SetIndex(int pos) { index = pos;  }
            public int GetIndex() { return index;  }

            //other getters
            public int getRouteLength() { return this.route.Count; }
            public ArrayList getRoute() { return this.route; }
            public double getBound() { return this.bound; }

            //method to set all items in a row to infinity
            private void infiniteRow(int row)
            {
                for (int i = 0; i < cities.Length; i++)
                {
                    matrix[row, i] = Double.PositiveInfinity;
                }
            }

            //method to set all items in a column to infinity
            private void infiniteCol(int col)
            {
                for (int i = 0; i < cities.Length; i++)
                {
                    matrix[i, col] = Double.PositiveInfinity;
                }
            }

            //method to find the minimum item in an array and return its position
            private int findMinOfArray(double[] arr)
            {
                int pos = 0;
                for (int i = 0; i < arr.Length; i++)
                {
                    if (arr[i] < arr[pos]) { pos = i; }
                }
                return pos;
            }

            //decrements every item in a row by the value "dec"
            private void decreaseMatrixRow(int row, double dec)
            {
                for (int i = 0; i < cities.Length; i++)
                {
                    if (matrix[row, i].Equals(Double.PositiveInfinity)) continue;
                    matrix[row, i] -= dec;
                }
            }

            //decrements every item in a column by the value "dec"
            private void decreaseMatrixCol(int col, double dec)
            {
                for (int i = 0; i < cities.Length; i++)
                {
                    if (matrix[i, col].Equals(Double.PositiveInfinity)) continue;
                    matrix[i, col] -= dec;
                }
            }

            /**
             * Reduces a matrix by making sure the minimum of each row and column is 0.
             * If the minimum is not 0, we add the minimum to the bound variable, and decrement everything in the row by the minimum.
             * Repeats for every column.
             */
            public void Reduce()
            {
                //temporary variables for each row and column.

                double[] row = new double[cities.Length]; 
                double[] col = new double[cities.Length];
                for(int i = 0; i < cities.Length; i++) //row
                {
                    for (int j = 0; j < cities.Length; j++) //column
                    {
                        row[j] = matrix[i, j]; //build the row
                    }
                    int minIndex = findMinOfArray(row); //find minimum of row
                    if (row[minIndex] != 0 && row[minIndex] != Double.PositiveInfinity) //if the minimum is not 0 and not infinity . . .
                    {
                        bound += row[minIndex]; //increase the bound
                        decreaseMatrixRow(i, row[minIndex]); //decrease the row
                    }

                }
                //repeat with columns
                for (int i = 0; i < cities.Length; i++) //column
                {
                    for (int j = 0; j < cities.Length; j++) //row
                    {
                        col[j] = matrix[j, i];
                    }
                    int minIndex = findMinOfArray(col);
                    if (col[minIndex] != 0 && col[minIndex] != Double.PositiveInfinity)
                    {
                        bound += col[minIndex];
                        decreaseMatrixCol(i, col[minIndex]);
                    }

                }
            }

            /**
             * returns a list of children nodes
             * each child node adds another unvisited city to its route
             */
            public List<bBNode> Expand()
            {
                List<bBNode> children = new List<bBNode>();
                for(int c = 0; c < cities.Length; c++){
                    if (used[c])
                    {
                        continue;
                    }
                    
                    bBNode child = new bBNode(this, c);
                    children.Add(child);
                }
                return children;
            }

            /** A Node is more than another Node if it has more routes. 
             * If they have the same routes, we want the route with the smaller bound*/
            public static bool operator > (bBNode sol1, bBNode sol2)
            {
                if (sol2 == null) return true; 
                if (sol1.getRouteLength() == sol2.getRouteLength())
                {
                    return sol1.getBound() < sol2.getBound(); // return smaller bound
                }
                else
                {
                    return sol1.getRouteLength() > sol2.getRouteLength(); //return longer route
                }
                
            }

            /** A Node is less than another Node if it has fewer routes. 
             * If they have the same routes, we want the route with the larger bound*/
            public static bool operator < (bBNode sol1, bBNode sol2)
            {
                if (sol2 == null) return true;
                if (sol1.getRouteLength() == sol2.getRouteLength())
                {
                    return sol1.getBound() > sol2.getBound(); //return larger bound
                }
                else
                {
                    return sol1.getRouteLength() < sol2.getRouteLength(); //returne smaller route
                }

            }
        }

        /// <summary>
        /// performs a Branch and Bound search of the state space of partial tours
        /// stops when time limit expires and uses BSSF as solution
        /// @returns results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)
        public string[] bBSolveProblem()
        {
            // counts the number of bssf updates / solutions found
            int count = 0; 

            // counts the number of pruned states
            int pruned = 0;

            // counts the number of total states
            int total_states = 0;

            //container for results
            string[] results = new string[3];

            //priority queue implementation
            HeapQueue myqueue = new HeapQueue();

            Route = new ArrayList();
            Stopwatch timer = new Stopwatch();
            timer.Start();
            Route.Clear();

            //generate greedy bssf as initial guess
            greedySolveProblem();
            
            //initialize a matrix with all the cities
            bBNode init = new bBNode(GetCities());

            /**
             * Starts at the first city (0).
             * It doesn't matter where we start because we are looking for a "loop" of cities.
             * And we don't want to repeatedly find the same solution by starting at a different city.
             */
            bBNode root = new bBNode(init, 0);
            bBNode next;

            //initialize queue
            myqueue.MakeQueue((int)Math.Pow(Cities.Length, 2), root);
            total_states++;
            
            //while the queue is not empty and we haven't hit the time limit . . . 
            while (!myqueue.IsEmpty() && timer.ElapsedMilliseconds < this.time_limit)
            {
                // take the top node in the queue
                next = (bBNode)myqueue.DeleteBestRoute();
                //if it has a better solution than the bssf, go deeper, otherwise, we prune it.
                if(next.getBound() < bssf.costOfRoute())
                {
                    // check if the route is finished
                    if (next.getRouteLength() == Cities.Length)
                    {
                        //We found a better solution!
                        bssf = new TSPSolution(next.getRoute());
                        count++;
                    }
                    else
                    {
                        // expand the node, and only insert the children that do better than the bssf
                        List<bBNode> children = next.Expand();
                        foreach (bBNode c in children)
                        {
                            total_states++;
                            if (c.getBound() < bssf.costOfRoute()){
                                myqueue.Insert(c);
                            }
                            else
                            {
                                pruned++;
                            }

                        }
                    }
                }
                else
                {
                    pruned++;
                }
            }
            timer.Stop();

            Console.WriteLine("**************************************");
            Console.WriteLine("# cities: " + Cities.Length);
            Console.WriteLine("# seed: "+ _seed);
            Console.WriteLine("RunningTime: " + timer.Elapsed);
            Console.WriteLine("Best tour: " + costOfBssf());
            Console.WriteLine("Max stored states: " + myqueue.getMaxStoredStates());
            Console.WriteLine("Number of Solutions: " + count);
            Console.WriteLine("Number total states: " + total_states);
            Console.WriteLine("Total pruned states: " + pruned);
         
            results[COST] = costOfBssf().ToString();                          // load results array
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            return results;
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        // These additional solver methods will be implemented as part of the group project.
        ////////////////////////////////////////////////////////////////////////////////////////////

        

        /// <summary>
        /// finds the greedy tour starting from each city and keeps the best (valid) one
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] greedySolveProblem()
        {
            // TODO: Add your implementation for a greedy solver here.
            int i, count = 0;
            string[] results = new string[3];
            
            Route = new ArrayList();
            Stopwatch timer = new Stopwatch();

            timer.Start();
            Route.Clear();
            for (i = 0; i < Cities.Length; i++)                            // Now build the route using the random permutation 
            {
                Route.Add(Cities[i]);
            }
            bssf = new TSPSolution(Route);

            for (i=0; i < Cities.Length; i++ ){
                City[] queue = GetCities(); //queue of cities not visited
                
                Route.Clear();
                City Prev = queue[i];
                queue[i] = null;
                Route.Add(Prev);
                               
                for (int j = 0; j < queue.Length; j++) // until we have reached all cities
                {
                    double minNeighborCost = double.PositiveInfinity;
                    int minNeighborIndex = -1;
                    for (int k = 0; k < queue.Length; k++) //find the minimum neighbor
                    {
                        if (queue[k] == null) continue;
                        if (Prev.costToGetTo(queue[k]) < minNeighborCost)
                        {
                            minNeighborCost = Prev.costToGetTo(queue[k]);
                            minNeighborIndex = k;
                        }
                    }
                    if (minNeighborIndex == -1)
                    {
                        break;
                    }
                    Prev = queue[minNeighborIndex];
                    queue[minNeighborIndex] = null;
                    Route.Add(Prev);
                }
                TSPSolution tsp = new TSPSolution(Route);
                count++; 
                if (tsp.costOfRoute() < bssf.costOfRoute())
                    bssf = tsp;
            }
            timer.Stop();

            results[COST] = costOfBssf().ToString();                          // load results array
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            return results;
        }

        public string[] fancySolveProblem()
        {
            string[] results = new string[3];

            // TODO: Add your implementation for your advanced solver here.

            results[COST] = "not implemented";    // load results into array here, replacing these dummy values
            results[TIME] = "-1";
            results[COUNT] = "-1";

            return results;
        }
        #endregion
    }

}
