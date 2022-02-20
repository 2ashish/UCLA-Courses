#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
//#include <sys/types.h>
#include <errno.h>
#include<sys/wait.h>

int main(int argc, char *argv[])
{
	if(argc<2){
		perror("invalid argument");
		exit(EINVAL);
	}
	//open two pipe
	int fd[2],ff[2];
	if (pipe(fd) == -1) {
    	int err = errno;
    	perror("pipe");
    	return err;
  	}
	if (pipe(ff) == -1) {
    	int err = errno;
    	perror("pipe");
    	return err;
  	}
	//no input for first process
	close(ff[1]);
	for(int i=1;i<argc;i++){
		int pid = fork();
		if (pid == -1) {
			int ret = errno;
			perror("fork failed");
    		return ret;
  		}
		if(pid==0){
			//child
			//printf("child i: %d\n",i);
			
			if(i%2==1){
				dup2(ff[0],STDIN_FILENO);
				if(i!=(argc-1))dup2(fd[1],STDOUT_FILENO);
				close(fd[0]);
				close(fd[1]);
				close(ff[0]);
			}
			else {
				dup2(fd[0],STDIN_FILENO);
				if(i!=(argc-1))dup2(ff[1],STDOUT_FILENO);
				close(fd[0]);
				close(ff[0]);
				close(ff[1]);
			}
			//execute ith argument/process
			int execlp_ret = execlp(argv[i],argv[i],NULL);
			if(execlp_ret==-1){
				execlp_ret = errno;
				perror("execlp failed");
				return execlp_ret;
			}
		}
		else{
			//parent
			int wait_status;
			wait(&wait_status);
			if(WIFEXITED(wait_status) && (WEXITSTATUS(wait_status)!=0)){
				perror("child process returned error");
				exit(WEXITSTATUS(wait_status));
			}
			//close write end of one pipe and create another alternatively
			if(i%2==1){
				close(fd[1]);
				close(ff[0]);
				if (pipe(ff) == -1) {
    				int err = errno;
    				perror("pipe");
    				return err;
  				}
			}
			else {
				close(fd[0]);
				close(ff[1]);
				if (pipe(fd) == -1) {
    				int err = errno;
    				perror("pipe");
    				return err;
  				}
			}
			
		}
	}
	return 0;
}
