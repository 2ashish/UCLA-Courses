#include <linux/module.h>
#include <linux/printk.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/sched.h> 

//global variable to store proc_entry
static struct proc_dir_entry *count_dir;

static int proc_count_show(struct seq_file *m, void *v)
{
	int i=0;
	struct task_struct *p;
	//counting number of processes and writing in file
	for_each_process(p){
		i++;
	}
	seq_printf(m, "%d\n", i);
	return 0;
}

static int __init proc_count_init(void)
{
	pr_info("proc_count: init\n");
	//creating proc entry
	count_dir = proc_create_single("count",0,NULL,proc_count_show);
	return 0;
}

static void __exit proc_count_exit(void)
{
	//removing proc entry
	pr_info("proc_count: exit\n");
	proc_remove(count_dir);
	return;
}
module_init(proc_count_init);
module_exit(proc_count_exit);

MODULE_AUTHOR("Ashish Kumar Singh");
MODULE_DESCRIPTION("Count number of processes running");
MODULE_LICENSE("GPL");
