# TODO

### Instructions for running the code
Running the tests is done with the command  ```make -B test```. To run the AI first run ```make -B ai``` which creates the ```ai``` binary. You need to run this with three flags. The first one is either ```-W``` or ```-B``` which specifies which AI should start. Then two flags with either ```-r``` for random or ```-s``` for one step ahead AI. The first flag is for the white AI and the second for the black AI.

Example: 

```make -B ai```

```./ai -W -r -s```


