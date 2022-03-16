// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !nacl,!plan9,!windows

package net

import (
	"bytes"
	"os"
	"reflect"
	"runtime"
	"syscall"
	"testing"
	"time"
)

func TestReadUnixgramWithUnnamedSocket(t *testing.T) {
	if !testableNetwork("unixgram") {
		t.Skip("unixgram test")
	}

	addr := testUnixAddr()
	la, err := ResolveUnixAddr("unixgram", addr)
	if err != nil {
		t.Fatal(err)
	}
	c, err := ListenUnixgram("unixgram", la)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		c.Close()
		os.Remove(addr)
	}()

	off := make(chan bool)
	data := [5]byte{1, 2, 3, 4, 5}
	go func() {
		defer func() { off <- true }()
		s, err := syscall.Socket(syscall.AF_UNIX, syscall.SOCK_DGRAM, 0)
		if err != nil {
			t.Error(err)
			return
		}
		defer syscall.Close(s)
		rsa := &syscall.SockaddrUnix{Name: addr}
		if err := syscall.Sendto(s, data[:], 0, rsa); err != nil {
			t.Error(err)
			return
		}
	}()

	<-off
	b := make([]byte, 64)
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	n, from, err := c.ReadFrom(b)
	if err != nil {
		t.Fatal(err)
	}
	if from != nil {
		t.Fatalf("unexpected peer address: %v", from)
	}
	if !bytes.Equal(b[:n], data[:]) {
		t.Fatalf("got %v; want %v", b[:n], data[:])
	}
}

func TestUnixgramZeroBytePayload(t *testing.T) {
	if !testableNetwork("unixgram") {
		t.Skip("unixgram test")
	}

	c1, err := newLocalPacketListener("unixgram")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(c1.LocalAddr().String())
	defer c1.Close()

	c2, err := Dial("unixgram", c1.LocalAddr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(c2.LocalAddr().String())
	defer c2.Close()

	for _, genericRead := range []bool{false, true} {
		n, err := c2.Write(nil)
		if err != nil {
			t.Fatal(err)
		}
		if n != 0 {
			t.Errorf("got %d; want 0", n)
		}
		c1.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		var b [1]byte
		var peer Addr
		if genericRead {
			_, err = c1.(Conn).Read(b[:])
		} else {
			_, peer, err = c1.ReadFrom(b[:])
		}
		switch err {
		case nil: // ReadFrom succeeds
			if peer != nil { // peer is connected-mode
				t.Fatalf("unexpected peer address: %v", peer)
			}
		default: // Read may timeout, it depends on the platform
			if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
				t.Fatal(err)
			}
		}
	}
}

func TestUnixgramZeroByteBuffer(t *testing.T) {
	if !testableNetwork("unixgram") {
		t.Skip("unixgram test")
	}
	// issue 4352: Recvfrom failed with "address family not
	// supported by protocol family" if zero-length buffer provided

	c1, err := newLocalPacketListener("unixgram")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(c1.LocalAddr().String())
	defer c1.Close()

	c2, err := Dial("unixgram", c1.LocalAddr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(c2.LocalAddr().String())
	defer c2.Close()

	b := []byte("UNIXGRAM ZERO BYTE BUFFER TEST")
	for _, genericRead := range []bool{false, true} {
		n, err := c2.Write(b)
		if err != nil {
			t.Fatal(err)
		}
		if n != len(b) {
			t.Errorf("got %d; want %d", n, len(b))
		}
		c1.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		var peer Addr
		if genericRead {
			_, err = c1.(Conn).Read(nil)
		} else {
			_, peer, err = c1.ReadFrom(nil)
		}
		switch err {
		case nil: // ReadFrom succeeds
			if peer != nil { // peer is connected-mode
				t.Fatalf("unexpected peer address: %v", peer)
			}
		default: // Read may timeout, it depends on the platform
			if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
				t.Fatal(err)
			}
		}
	}
}

func TestUnixgramAutobind(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("autobind is linux only")
	}

	laddr := &UnixAddr{Name: "", Net: "unixgram"}
	c1, err := ListenUnixgram("unixgram", laddr)
	if err != nil {
		t.Fatal(err)
	}
	defer c1.Close()

	// retrieve the autobind address
	autoAddr := c1.LocalAddr().(*UnixAddr)
	if len(autoAddr.Name) <= 1 {
		t.Fatalf("invalid autobind address: %v", autoAddr)
	}
	if autoAddr.Name[0] != '@' {
		t.Fatalf("invalid autobind address: %v", autoAddr)
	}

	c2, err := DialUnix("unixgram", nil, autoAddr)
	if err != nil {
		t.Fatal(err)
	}
	defer c2.Close()

	if !reflect.DeepEqual(c1.LocalAddr(), c2.RemoteAddr()) {
		t.Fatalf("expected autobind address %v, got %v", c1.LocalAddr(), c2.RemoteAddr())
	}
}

func TestUnixAutobindClose(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("autobind is linux only")
	}

	laddr := &UnixAddr{Name: "", Net: "unix"}
	ln, err := ListenUnix("unix", laddr)
	if err != nil {
		t.Fatal(err)
	}
	ln.Close()
}

func TestUnixgramWrite(t *testing.T) {
	if !testableNetwork("unixgram") {
		t.Skip("unixgram test")
	}

	addr := testUnixAddr()
	laddr, err := ResolveUnixAddr("unixgram", addr)
	if err != nil {
		t.Fatal(err)
	}
	c, err := ListenPacket("unixgram", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(addr)
	defer c.Close()

	testUnixgramWriteConn(t, laddr)
	testUnixgramWritePacketConn(t, laddr)
}

func testUnixgramWriteConn(t *testing.T, raddr *UnixAddr) {
	c, err := Dial("unixgram", raddr.String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	b := []byte("CONNECTED-MODE SOCKET")
	if _, err := c.(*UnixConn).WriteToUnix(b, raddr); err == nil {
		t.Fatal("should fail")
	} else if err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("should fail as ErrWriteToConnected: %v", err)
	}
	if _, err = c.(*UnixConn).WriteTo(b, raddr); err == nil {
		t.Fatal("should fail")
	} else if err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("should fail as ErrWriteToConnected: %v", err)
	}
	if _, _, err = c.(*UnixConn).WriteMsgUnix(b, nil, raddr); err == nil {
		t.Fatal("should fail")
	} else if err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("should fail as ErrWriteToConnected: %v", err)
	}
	if _, err := c.Write(b); err != nil {
		t.Fatal(err)
	}
}

func testUnixgramWritePacketConn(t *testing.T, raddr *UnixAddr) {
	addr := testUnixAddr()
	c, err := ListenPacket("unixgram", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(addr)
	defer c.Close()

	b := []byte("UNCONNECTED-MODE SOCKET")
	if _, err := c.(*UnixConn).WriteToUnix(b, raddr); err != nil {
		t.Fatal(err)
	}
	if _, err := c.WriteTo(b, raddr); err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.(*UnixConn).WriteMsgUnix(b, nil, raddr); err != nil {
		t.Fatal(err)
	}
	if _, err := c.(*UnixConn).Write(b); err == nil {
		t.Fatal("should fail")
	}
}

func TestUnixConnLocalAndRemoteNames(t *testing.T) {
	if !testableNetwork("unix") {
		t.Skip("unix test")
	}

	handler := func(ls *localServer, ln Listener) {}
	for _, laddr := range []string{"", testUnixAddr()} {
		laddr := laddr
		taddr := testUnixAddr()
		ta, err := ResolveUnixAddr("unix", taddr)
		if err != nil {
			t.Fatal(err)
		}
		ln, err := ListenUnix("unix", ta)
		if err != nil {
			t.Fatal(err)
		}
		ls, err := (&streamListener{Listener: ln}).newLocalServer()
		if err != nil {
			t.Fatal(err)
		}
		defer ls.teardown()
		if err := ls.buildup(handler); err != nil {
			t.Fatal(err)
		}

		la, err := ResolveUnixAddr("unix", laddr)
		if err != nil {
			t.Fatal(err)
		}
		c, err := DialUnix("unix", la, ta)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			c.Close()
			if la != nil {
				defer os.Remove(laddr)
			}
		}()
		if _, err := c.Write([]byte("UNIXCONN LOCAL AND REMOTE NAME TEST")); err != nil {
			t.Fatal(err)
		}

		switch runtime.GOOS {
		case "android", "linux":
			if laddr == "" {
				laddr = "@" // autobind feature
			}
		}
		var connAddrs = [3]struct{ got, want Addr }{
			{ln.Addr(), ta},
			{c.LocalAddr(), &UnixAddr{Name: laddr, Net: "unix"}},
			{c.RemoteAddr(), ta},
		}
		for _, ca := range connAddrs {
			if !reflect.DeepEqual(ca.got, ca.want) {
				t.Fatalf("got %#v, expected %#v", ca.got, ca.want)
			}
		}
	}
}

func TestUnixgramConnLocalAndRemoteNames(t *testing.T) {
	if !testableNetwork("unixgram") {
		t.Skip("unixgram test")
	}

	for _, laddr := range []string{"", testUnixAddr()} {
		laddr := laddr
		taddr := testUnixAddr()
		ta, err := ResolveUnixAddr("unixgram", taddr)
		if err != nil {
			t.Fatal(err)
		}
		c1, err := ListenUnixgram("unixgram", ta)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			c1.Close()
			os.Remove(taddr)
		}()

		var la *UnixAddr
		if laddr != "" {
			if la, err = ResolveUnixAddr("unixgram", laddr); err != nil {
				t.Fatal(err)
			}
		}
		c2, err := DialUnix("unixgram", la, ta)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			c2.Close()
			if la != nil {
				defer os.Remove(laddr)
			}
		}()

		switch runtime.GOOS {
		case "android", "linux":
			if laddr == "" {
				laddr = "@" // autobind feature
			}
		}

		var connAddrs = [4]struct{ got, want Addr }{
			{c1.LocalAddr(), ta},
			{c1.RemoteAddr(), nil},
			{c2.LocalAddr(), &UnixAddr{Name: laddr, Net: "unixgram"}},
			{c2.RemoteAddr(), ta},
		}
		for _, ca := range connAddrs {
			if !reflect.DeepEqual(ca.got, ca.want) {
				t.Fatalf("got %#v; want %#v", ca.got, ca.want)
			}
		}
	}
}

// forceGoDNS forces the resolver configuration to use the pure Go resolver
// and returns a fixup function to restore the old settings.
func forceGoDNS() func() {
	c := systemConf()
	oldGo := c.netGo
	oldCgo := c.netCgo
	fixup := func() {
		c.netGo = oldGo
		c.netCgo = oldCgo
	}
	c.netGo = true
	c.netCgo = false
	return fixup
}

// forceCgoDNS forces the resolver configuration to use the cgo resolver
// and returns true to indicate that it did so.
// (On non-Unix systems forceCgoDNS returns false.)
func forceCgoDNS() bool {
	c := systemConf()
	c.netGo = false
	c.netCgo = true
	return true
}
