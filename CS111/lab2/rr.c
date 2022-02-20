#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/queue.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef uint32_t u32;
typedef int32_t i32;

struct process {
  u32 pid;
  u32 arrival_time;
  u32 burst_time;

  TAILQ_ENTRY(process) pointers;

  /* Additional fields here */
  u32 rem_time;
  u32 start_time;
  u32 end_time;
  bool ran_before;
  /* End of "Additional fields here" */
};

TAILQ_HEAD(process_list, process);

u32 next_int(const char **data, const char *data_end) {
  u32 current = 0;
  bool started = false;
  while (*data != data_end) {
    char c = **data;

    if (c < 0x30 || c > 0x39) {
      if (started) {
	return current;
      }
    }
    else {
      if (!started) {
	current = (c - 0x30);
	started = true;
      }
      else {
	current *= 10;
	current += (c - 0x30);
      }
    }

    ++(*data);
  }

  printf("Reached end of file while looking for another integer\n");
  exit(EINVAL);
}

u32 next_int_from_c_str(const char *data) {
  char c;
  u32 i = 0;
  u32 current = 0;
  bool started = false;
  while ((c = data[i++])) {
    if (c < 0x30 || c > 0x39) {
      exit(EINVAL);
    }
    if (!started) {
      current = (c - 0x30);
      started = true;
    }
    else {
      current *= 10;
      current += (c - 0x30);
    }
  }
  return current;
}

void init_processes(const char *path,
                    struct process **process_data,
                    u32 *process_size)
{
  int fd = open(path, O_RDONLY);
  if (fd == -1) {
    int err = errno;
    perror("open");
    exit(err);
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    int err = errno;
    perror("stat");
    exit(err);
  }

  u32 size = st.st_size;
  const char *data_start = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data_start == MAP_FAILED) {
    int err = errno;
    perror("mmap");
    exit(err);
  }

  const char *data_end = data_start + size;
  const char *data = data_start;
  

  *process_size = next_int(&data, data_end);

  *process_data = calloc(sizeof(struct process), *process_size);
  if (*process_data == NULL) {
    int err = errno;
    perror("calloc");
    exit(err);
  }

  for (u32 i = 0; i < *process_size; ++i) {
    (*process_data)[i].pid = next_int(&data, data_end);
    (*process_data)[i].arrival_time = next_int(&data, data_end);
    (*process_data)[i].burst_time = next_int(&data, data_end);
  }
  
  munmap((void *)data, size);
  close(fd);
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    return EINVAL;
  }
  struct process *data;
  u32 size;
  init_processes(argv[1], &data, &size);

  u32 quantum_length = next_int_from_c_str(argv[2]);

  struct process_list list;
  TAILQ_INIT(&list);

  u32 total_waiting_time = 0;
  u32 total_response_time = 0;

  /* Your code here */
  struct process *cur, *cur_run;
  for(u32 i=0;i<size;i++){
      cur = &data[i];
      cur->rem_time = cur->burst_time;
      cur->start_time = 0;
      cur->end_time = 0;
      cur->ran_before = false;
  }

  u32 cur_time=0,cur_quant=0, num_process_finished=0;
  bool finished=false,running=false;
  //base case
  for(u32 i=0;i<size;i++){
      cur = &data[i];
      if(cur->arrival_time == 0)TAILQ_INSERT_TAIL(&list, cur,pointers);
  }

  while(!finished){
    if(!running && !TAILQ_EMPTY(&list)){
      cur_run = TAILQ_FIRST(&list);
      //printf("running %d %d\n",cur_run->pid, cur_time);
      TAILQ_REMOVE(&list,cur_run,pointers);
      if(!cur_run->ran_before){
        cur_run->ran_before = true;
        cur_run->start_time = cur_time;
      }
      running =true;
      cur_quant=0;
    }
    cur_time++;
    for(u32 i=0;i<size;i++){
      cur = &data[i];
      if(cur->arrival_time == cur_time)TAILQ_INSERT_TAIL(&list, cur,pointers);
    }
    if(running){
      cur_quant++;
      cur_run->rem_time--;
      if(cur_run->rem_time == 0){
        cur_run->end_time = cur_time;
        num_process_finished++;
        total_waiting_time+= cur_run->end_time - cur_run->arrival_time - cur_run->burst_time;
        total_response_time+= cur_run->start_time - cur_run->arrival_time;
        //printf("finished %d %d\n",cur_run->pid, cur_time);
        running=false;
        cur_quant=0;
      }
      else {
        if(cur_quant==quantum_length){
          cur_quant=0;
          running=false;
          TAILQ_INSERT_TAIL(&list, cur_run,pointers);
        }
      }
    }
    if(num_process_finished==size)finished=true;

  }
  /*TAILQ_FOREACH(p, &list, pointers) {
      printf(" %d ", p->pid);
  }*/
  /* End of "Your code here" */

  printf("Average waiting time: %.2f\n", (float) total_waiting_time / (float) size);
  printf("Average response time: %.2f\n", (float) total_response_time / (float) size);

  free(data);
  return 0;
}
